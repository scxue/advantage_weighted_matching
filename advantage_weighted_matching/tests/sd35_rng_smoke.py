"""Real pretrained SD3.5: verify setup RNG and legacy-vs-logical rollout inputs."""
import json
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from config.dgx_awm import geneval_sd3_no_cfg_4nodes
from scripts.train_sd3_awm import initialize_pipeline, compute_text_embeddings, GenevalPromptDataset
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.resume import Legacy32Layout, LogicalState, capture_rng, restore_rng


def main():
    acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=18)
    config = geneval_sd3_no_cfg_4nodes()
    set_seed(config.seed, device_specific=True)
    pipeline, encoders, tokenizers, transformer, parameters, ema, refs = initialize_pipeline(config, acc)
    initial = capture_rng(acc.device)
    optimizer = torch.optim.AdamW(parameters, lr=config.train.learning_rate)
    neg, neg_pool = compute_text_embeddings([""], encoders, tokenizers, max_sequence_length=128, device=acc.device)
    transformer, optimizer = acc.prepare(transformer, optimizer)
    after = capture_rng(acc.device)
    assert torch.equal(initial["cpu"], after["cpu"]), "Post-init CPU RNG changed"
    assert torch.equal(initial["cuda"], after["cuda"]), "Post-init CUDA RNG changed"
    torch.empty((), dtype=torch.int64).random_()  # Original train iterator seed.
    dataset = GenevalPromptDataset(config.dataset)
    layout = Legacy32Layout(len(dataset), 32, acc.process_index)
    indices = layout.indices(2, 0, acc.process_index)
    prompts, metadata = dataset.collate_fn([dataset[i] for i in indices])
    embeds, pooled = compute_text_embeddings(prompts, encoders, tokenizers, max_sequence_length=128, device=acc.device)
    pipeline.transformer.eval()
    before = capture_rng(acc.device)
    def sample():
        with torch.no_grad():
            return pipeline_with_logprob(
                pipeline, prompt_embeds=embeds, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg.repeat(6, 1, 1), negative_pooled_prompt_embeds=neg_pool.repeat(6, 1),
                num_inference_steps=14, guidance_scale=1.0, output_type="pt", return_dict=False,
                height=512, width=512, noise_level=.4, sde_frac=0., use_sa_solver=True,
            )
    expected_images, expected_latents = sample()
    expected_rng = capture_rng(acc.device)
    logical = LogicalState(layout, acc.device, {acc.process_index: before}, {acc.process_index: ema})
    with logical.use(acc.process_index):
        images, latents = sample()
    assert torch.equal(expected_images, images), "Real rollout images differ"
    assert all(torch.equal(a, b) for a, b in zip(expected_latents, latents)), "Real rollout latents differ"
    assert torch.equal(expected_rng["cuda"], logical.rngs[acc.process_index]["cuda"])
    print(json.dumps({"status": "REAL_SD35_PASS", "rank": acc.process_index,
                      "steps": 14, "batch": 6, "resolution": 512, "sa_solver": True,
                      "exact_images_and_latents": True}), flush=True)
    acc.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
