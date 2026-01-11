from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.torch_utils import randn_tensor
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import wandb
from functools import partial
import tqdm
import tempfile
import itertools
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size per gpu
        self.k = k                    # number of repetitions per sample
        self.num_replicas = num_replicas  # total number of gpus
        self.rank = rank              # current gpu rank
        self.seed = seed              # random seed for synchronization
        
        # calculate the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # number of unique samples
        self.epoch=0

    def __iter__(self):
        while True:
            # generate a deterministic random sequence to ensure all gpus are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # print('epoch', self.epoch)
            # randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            # print(self.rank, 'indices', indices)
            # repeat each sample k times to generate total samples n*b
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            # print(self.rank, 'shuffled_samples', shuffled_samples)
            # split samples to each gpu
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            # print(self.rank, 'per_card_samples', per_card_samples[self.rank])
            # return the sample indices for the current gpu
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # used to synchronize random state across epochs

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def set_adapter_and_freeze_params(transformer, adapter_name):
    transformer.module.set_adapter(adapter_name)
    for name, param in transformer.named_parameters():
        if "learner" in name:
            param.requires_grad_(True)
        elif "ref" in name:
            param.requires_grad_(False)

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
        all_correct_ratio: Proportion of prompts with zero std and mean reward 1.0.
        all_wrong_ratio: Proportion of prompts with zero std and mean reward 0.0.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate mean for each group (Added for checking correctness)
    prompt_means = np.array([np.mean(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_mask = (prompt_std_devs == 0)
    zero_std_count = np.count_nonzero(zero_std_mask)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    # Calculate the ratio of all correct (zero std + mean 1.0)
    all_correct_count = np.count_nonzero(zero_std_mask & np.isclose(prompt_means, 1.0))
    all_correct_ratio = all_correct_count / len(prompt_std_devs)
    
    # Calculate the ratio of all wrong (zero std + mean 0.0)
    all_wrong_count = np.count_nonzero(zero_std_mask & np.isclose(prompt_means, 0.0))
    all_wrong_ratio = all_wrong_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean(), all_correct_ratio, all_wrong_ratio
    

def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_log_prob_awm(transformer, pipeline, sample, timestep, embeds, pooled_embeds, config, noised_latents, clean_latents, random_noise, weighting='Uniform'):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([noised_latents] * 2),
            timestep=torch.cat([timestep.view(-1)*1000] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=noised_latents,
            timestep=timestep.view(-1)*1000,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    sigma = timestep

    std_dev_t = torch.sqrt(sigma / (1 - torch.clamp(sigma, 0, 0.99)))*0.7
    model_output = noise_pred.double()
    log_prob = -(model_output - (random_noise.double() - clean_latents.double()))**2
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    if weighting == 't':
        log_prob = log_prob * timestep.view(-1)
    elif weighting == 't**2':
        log_prob = log_prob * timestep.view(-1)**2
    elif weighting == 'Uniform':
        log_prob = log_prob
    elif weighting == 'huber':
        log_prob = -(torch.sqrt(-log_prob + 1e-10) - 1e-5)* timestep.view(-1)
    elif weighting == 'ghuber':
        log_prob = -(torch.pow(-log_prob + 1e-10, config.train.ghuber_power) - torch.pow(torch.tensor(1e-10, device=log_prob.device, dtype=log_prob.dtype), config.train.ghuber_power)) * timestep.view(-1) / config.train.ghuber_power
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")
    return log_prob, model_output, std_dev_t

def logit_normal_shifted_sampler(shape, m=0.0, s=1.0, shift=3.0, device='cpu'):
    """
    Args:
        shape (torch.Size or tuple): The shape of the output tensor.
        m (float): The mean (location parameter) of the normal distribution.
        s (float): The standard deviation (scale parameter) of the normal distribution.
        shift (float): The shift parameter of the Logit-Normal distribution.
        device (str): The device to create the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor with values sampled from the Logit-Normal distribution, where the values are in the range (0, 1).
    """
    u_standard = torch.randn(shape, device=device)
    
    u = u_standard * s + m

    t = torch.sigmoid(u)
    
    t = shift * t / (1 + (shift - 1) * t)

    return t
    

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters):
    def log_metrics(data: dict, step=None):
        """
        Unified logger: 
        - accelerator.log 用于保持你原来的逻辑 (不会上传)
        - wandb.log 用于实际记录
        """
        accelerator.log(data, step=step)    # 本地统计逻辑
        if accelerator.is_main_process:
            wandb.log(data, step=step)      # 上传到 byted-wandb
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval non-EMA: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # the last batch may not have batch_size samples
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, latents = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    height=config.resolution,
                    width=config.resolution, 
                    determistic=True,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # use new index
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            log_metrics(
                {
                    "eval_images_non_ema": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_non_ema_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

        sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
        sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

        # test_dataloader = itertools.islice(test_dataloader, 2)
        all_rewards = defaultdict(list)
        for test_batch in tqdm(
                test_dataloader,
                desc="Eval EMA: ",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
            prompts, prompt_metadata = test_batch
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            # the last batch may not have batch_size samples
            if len(prompt_embeds)<len(sample_neg_prompt_embeds):
                sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
                sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
            with autocast():
                with torch.no_grad():
                    images, latents = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        return_dict=False,
                        height=config.resolution,
                        width=config.resolution, 
                        determistic=True,
                    )
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
            # yield to to make sure reward computation starts
            time.sleep(0)
            rewards, reward_metadata = rewards.result()

            for key, value in rewards.items():
                rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                all_rewards[key].append(rewards_gather)
        
        last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
        last_batch_prompt_ids = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
        last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
            last_batch_prompt_ids_gather, skip_special_tokens=True
        )
        last_batch_rewards_gather = {}
        for key, value in rewards.items():
            last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

        all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
        if accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(last_batch_images_gather))
                # sample_indices = random.sample(range(len(images)), num_samples)
                sample_indices = range(num_samples)
                for idx, index in enumerate(sample_indices):
                    image = last_batch_images_gather[index]
                    pil = Image.fromarray(
                        (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # use new index
                sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
                sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
                for key, value in all_rewards.items():
                    print(key, value.shape)
                log_metrics(
                    {
                        "eval_images_ema": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                            )
                            for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                        **{f"eval_reward_ema_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                    },
                    step=global_step,
                )

        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    if accelerator.is_main_process:
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora_non_ema = os.path.join(save_root, "lora_non_ema")
        os.makedirs(save_root_lora_non_ema, exist_ok=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora_non_ema)
        save_root_lora_ema = os.path.join(save_root, "lora_ema")
        os.makedirs(save_root_lora_ema, exist_ok=True)
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
            unwrap_model(transformer, accelerator).save_pretrained(save_root_lora_ema)
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    unique_id = datetime.datetime.now().strftime("%m.%d_%H")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with=None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps, # we deal with gradient accumulation logic in the given configs
    )
    if accelerator.is_main_process:
        # accelerator.init_trackers(
        #     project_name="sxue_RL",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
        wandb.init(
            project="sxue_RL",
            name=config.run_name,
            config=config.to_dict(),
        )
    def log_metrics(data: dict, step=None):
        """
        Unified logger: 
        - accelerator.log 用于保持你原来的逻辑 (不会上传)
        - wandb.log 用于实际记录
        """
        accelerator.log(data, step=step)    # 本地统计逻辑
        if accelerator.is_main_process:
            wandb.log(data, step=step)      # 上传到 byted-wandb


    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move transformer, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        # pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
        pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        if config.lora_type == "attn":
            target_modules = [
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ]
        elif config.lora_type == "ffn":
            target_modules = [
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        else:
            raise NotImplementedError("Only attn and ffn are supported with lora_type")
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # after using PeftModel.from_pretrained to load, all parameters' requires_grad are False, need to set_adapter to make adapter parameters gradient True
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=config.train.ema_decay, update_step_interval=config.train.ema_update_step_interval, device=accelerator.device)

    # add EMA lora branch to maintain an adaptive KL regularization
    transformer.add_adapter("ema", transformer_lora_config)
    transformer.set_adapter("ema")
    ema_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("default")

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, ema_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.train.decay_steps,
        eta_min=config.train.learning_rate / 3,
    )

    constant_scheduler = ConstantLR(
        optimizer,
        factor= 1/3,
        total_iters=1000000 - config.train.decay_steps
    )

    lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[cosine_scheduler, constant_scheduler],
    milestones=[config.train.decay_steps]
)

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # create an infinite loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,  # your k value
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # create a DataLoader, note that here we don't need to shuffle, which is controlled by the Sampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )
        # create a normal DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        # create an infinite loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,  # your k value
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # create a DataLoader, note that here we don't need to shuffle, which is controlled by the Sampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        # create a normal DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per epoch = {samples_per_epoch // total_train_batch_size}"
    )

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    global_step = 0
    if config.train.lora_path:
        parts = config.train.lora_path.split('/')
        checkpoint_folder = parts[-2]
        step_str = checkpoint_folder.split('-')[-1]
        global_step = int(step_str)
        first_epoch = global_step // 2 + 3
        if accelerator.is_main_process:
            print(f"global_step: {global_step}, first_epoch: {first_epoch}")
    train_iter = iter(train_dataloader)

    for epoch in range(first_epoch, config.num_epochs):
        epoch_start_time = time.time()
        
        #################### SAMPLING ####################
        sampling_start_time = time.time()
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            if i==0 and epoch % config.eval_freq == 0 and epoch>0:
                eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters)
            if i==0 and epoch % config.save_freq == 0 and epoch>0 and accelerator.is_main_process:
                save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            # this is intentional, because the group size collected in the first two epochs has a bug, after two epochs, the group_size stabilizes to the specified value
            if epoch < 2:
                continue
            # sample
            if config.train.off_policy:
                transformer.module.set_adapter('ema')
            with autocast():
                with torch.no_grad():
                    images, latents = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        return_dict=False,
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        sde_frac=config.sample.sde_frac,
                        use_sa_solver=config.train.use_sa_solver,
                )
            if config.train.off_policy:
                transformer.module.set_adapter('default')

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "rewards": rewards,
                }
            )

        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time
        
        if epoch < 2:
            continue
        
        reward_start_time = time.time()
        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
        reward_end_time = time.time()
        reward_time = reward_end_time - reward_start_time

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # use new index

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                log_metrics(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1) # DO NOT use kl_reward
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        log_metrics(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                "sampling_time": sampling_time,
                "reward_time": reward_time,
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean, positive_zero_std_ratio, negative_zero_std_ratio = calculate_zero_std_ratio(prompts, gathered_rewards)

            log_metrics(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "positive_zero_std_ratio": positive_zero_std_ratio,
                    "negative_zero_std_ratio": negative_zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                },
                step=global_step,
            )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        log_metrics(
            {
                "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
            },
            step=global_step,
        )
        # Filter out samples where the entire time dimension of advantages is zero
        # samples = {k: v[mask] for k, v in samples.items()}
        # here we do not filter out samples where the entire time dimension of advantages is zero
        # to avoid the conflict with gradient accumulation in our logic

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps


        #################### TRAINING ####################
        training_start_time = time.time()
        # shuffle samples along batch dimension
        perm = torch.randperm(total_batch_size, device=accelerator.device)
        samples = {k: v[perm] for k, v in samples.items()}

        # shuffle along time dimension independently for each sample
        perms = torch.stack(
            [
                torch.arange(num_timesteps, device=accelerator.device)
                for _ in range(total_batch_size)
            ]
        )
        for key in ["timesteps", "latents", "next_latents"]:
            samples[key] = samples[key][
                torch.arange(total_batch_size, device=accelerator.device)[:, None],
                perms,
            ]

        # rebatch for training
        # here we use config.train.num_batches_per_epoch instead of config.sample.num_batches_per_epoch in our logic
        samples_batched = {
            k: v.reshape(-1, total_batch_size//config.train.num_batches_per_epoch, *v.shape[1:])
            for k, v in samples.items()
        }

        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        # train
        pipeline.transformer.train()
        info = defaultdict(list)

        # IMPORTANT: calculate old log_probs only for those off-policy samples
        for i, sample in tqdm(
            list(enumerate(samples_batched)),
            desc=f"Epoch {epoch}: Calculating old log_probs; off-policy sample only",
            position=0,
            disable=not accelerator.is_local_main_process,
        ):
            if config.train.cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                )
                pooled_embeds = torch.cat(
                    [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                )
            else:
                embeds = sample["prompt_embeds"]
                pooled_embeds = sample["pooled_prompt_embeds"]


            if config.time_type == 'logit_normal':
                # continuous t
                # same t with stratified sampling
                T = config.train.train_timesteps
                B = config.train.batch_size

                base_uniform_vector = (torch.arange(T, device=accelerator.device) + torch.rand(T, device=accelerator.device)) / T

                normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
                u_standard_vector = normal_dist.icdf(torch.clamp(base_uniform_vector, 1e-7, 1-1e-7))
                u_standard_vector = normal_dist.icdf(base_uniform_vector)

                # 3. shuffle this vector randomly
                u_standard_shuffled = u_standard_vector[torch.randperm(T, device=accelerator.device)]

                m, s, shift = 0.0, 1.0, config.time_shift
                u = u_standard_shuffled * s + m
                t_vector = torch.sigmoid(u)
                t_vector_shifted = shift * t_vector / (1 + (shift - 1) * t_vector)

                timesteps = torch.repeat_interleave(t_vector_shifted, repeats=B)
                timesteps = torch.clamp(timesteps, min=0.01)
                timesteps = timesteps.view(-1, 1, 1, 1)

            elif config.time_type == 'uniform':
                global_lower, global_upper = 0.20, 1.0
                rand_u = torch.rand(config.train.train_timesteps, config.train.batch_size, device=accelerator.device)
                normalized_matrix = (torch.arange(config.train.train_timesteps, device=accelerator.device).unsqueeze(1) + rand_u) / config.train.train_timesteps
                
                matrix = global_lower + normalized_matrix * (global_upper - global_lower)
                timesteps = torch.gather(matrix, 0, torch.rand_like(matrix).argsort(dim=0)).flatten()
                timesteps = config.time_shift * timesteps / (1 + (config.time_shift - 1) * timesteps)
                timesteps = timesteps.view(-1,1,1,1)

            elif config.time_type == 'discrete':
                boundaries = torch.linspace(0, int(config.sample.num_steps * config.train.timestep_fraction), steps=config.train.train_timesteps + 1, device=accelerator.device)
                lower_bounds = boundaries[:-1].long()
                upper_bounds = boundaries[1:].long()
                rand_u = torch.rand(config.train.train_timesteps, device=accelerator.device)
                t_indices = lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()

                train_totalindex = torch.repeat_interleave(t_indices, repeats=config.train.batch_size) # [12]
                timesteps = sample["timesteps"][0][train_totalindex].view(-1,1,1,1) / 1000 # [12, 1, 1, 1]
            elif config.time_type == 'discrete_with_init':
                init_index = torch.tensor([0], device=accelerator.device, dtype=torch.long)
                num_remaining = config.train.train_timesteps - 1
                if num_remaining > 0:
                    max_step_idx = int(config.sample.num_steps * config.train.timestep_fraction)
                    boundaries = torch.linspace(1, max_step_idx, steps=num_remaining + 1, device=accelerator.device)
                    lower_bounds = boundaries[:-1].long()
                    upper_bounds = boundaries[1:].long()                
                    rand_u = torch.rand(num_remaining, device=accelerator.device)
                    remaining_indices = lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()
                    t_indices = torch.cat([init_index, remaining_indices])
                else:
                    t_indices = init_index

                train_totalindex = torch.repeat_interleave(t_indices, repeats=config.train.batch_size) # [12]
                timesteps = sample["timesteps"][0][train_totalindex].view(-1,1,1,1) / 1000 # [12, 1, 1, 1]
            elif config.time_type == 'discrete_wo_init':
                boundaries = torch.linspace(1, int(config.sample.num_steps * config.train.timestep_fraction), steps=config.train.train_timesteps + 1, device=accelerator.device)
                lower_bounds = boundaries[:-1].long()
                upper_bounds = boundaries[1:].long()
                rand_u = torch.rand(config.train.train_timesteps, device=accelerator.device)
                t_indices = lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()

                train_totalindex = torch.repeat_interleave(t_indices, repeats=config.train.batch_size) # [12]
                timesteps = sample["timesteps"][0][train_totalindex].view(-1,1,1,1) / 1000 # [12, 1, 1, 1]
            else:
                raise NotImplementedError("Only logit_normal, uniform, and discrete are supported with time_type")

            clean_latents = sample["next_latents"][:, -1]
            
            # repeat embeds and pooled_embeds for batched sampling and training
            embeds = torch.cat([embeds] * config.train.train_timesteps, dim = 0)
            pooled_embeds = torch.cat([pooled_embeds] * config.train.train_timesteps, dim = 0)
            clean_latents = torch.cat([clean_latents] * config.train.train_timesteps, dim = 0)
            train_random_noise = torch.cat([randn_tensor(
                                    sample["latents"][:, 0].shape,
                                    generator=None,
                                    device=accelerator.device,
                                    dtype=sample["latents"][:, 0].dtype,
                                ) for _ in range(config.train.train_timesteps)], dim = 0)


            sample['timesteps'] = timesteps # [12, 1, 1, 1]
            sample['train_random_noise'] = train_random_noise # [12, 16, 64, 64]
            sample['clean_latents'] = clean_latents # [12, 16, 64, 64]
            # equivalent to i < config.sample.num_batches_per_epoch // ( config.sample.num_batches_per_epoch // config.train.gradient_accumulation_steps )
            if not config.train.off_policy:
                if i < config.train.gradient_accumulation_steps:
                    continue

            timesteps = sample["timesteps"]
            noised_latents = (1-timesteps) * sample['clean_latents'] + timesteps * sample['train_random_noise']
            if config.train.off_policy:
                transformer.module.set_adapter('ema')
            with autocast():
                with torch.no_grad():
                    log_prob, _, _ = compute_log_prob_awm(transformer, pipeline, sample, timesteps, embeds, pooled_embeds, config, noised_latents, sample['clean_latents'], sample['train_random_noise'], weighting=config.train.weighting)
            sample['old_log_probs'] = log_prob.reshape(config.train.train_timesteps, config.train.batch_size)
            if config.train.off_policy:
                transformer.module.set_adapter('default')

        for i, sample in tqdm(
            list(enumerate(samples_batched)),
            desc=f"Epoch {epoch}: training",
            position=0,
            disable=not accelerator.is_local_main_process,
        ):
            if config.train.cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                )
                pooled_embeds = torch.cat(
                    [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                )
            else:
                embeds = sample["prompt_embeds"]
                pooled_embeds = sample["pooled_prompt_embeds"]


            #  repeat embeds and pooled_embeds for batched sampling and training
            embeds = torch.cat([embeds] * config.train.train_timesteps, dim = 0)
            pooled_embeds = torch.cat([pooled_embeds] * config.train.train_timesteps, dim = 0)

            with accelerator.accumulate(transformer):

                timesteps = sample["timesteps"]
                noised_latents = (1-timesteps) * sample['clean_latents'] + timesteps * sample['train_random_noise']
                
                with autocast():
                    log_prob, model_output, std_dev_t = compute_log_prob_awm(transformer, pipeline, sample, timesteps, embeds, pooled_embeds, config, noised_latents, sample['clean_latents'], sample['train_random_noise'], weighting=config.train.weighting)
                    if config.train.beta > 0:
                        with torch.no_grad():
                            with transformer.module.disable_adapter():
                                log_prob_ref, model_output_ref, std_dev_t_ref = compute_log_prob_awm(transformer, pipeline, sample, timesteps, embeds, pooled_embeds, config, noised_latents, sample['clean_latents'], sample['train_random_noise'], weighting=config.train.weighting)
                    if config.train.ema_beta > 0:
                        transformer.module.set_adapter('ema')
                        with torch.no_grad():
                            _, model_output_ema, _ = compute_log_prob_awm(transformer, pipeline, sample, timesteps, embeds, pooled_embeds, config, noised_latents, sample['clean_latents'], sample['train_random_noise'], weighting=config.train.weighting)
                        transformer.module.set_adapter('default')

                if not config.train.off_policy:
                    if i < config.train.gradient_accumulation_steps:
                        sample['old_log_probs'] = log_prob.reshape(config.train.train_timesteps, config.train.batch_size).detach()
                sample['log_probs'] = log_prob.reshape(config.train.train_timesteps, config.train.batch_size)
                sample['ref_log_probs'] = log_prob_ref.reshape(config.train.train_timesteps, config.train.batch_size)
                sample['model_output'] = model_output.reshape(config.train.train_timesteps, config.train.batch_size, *model_output.shape[1:])
                sample['model_output_ref'] = model_output_ref.reshape(config.train.train_timesteps, config.train.batch_size, *model_output_ref.shape[1:])
                sample['model_output_ema'] = model_output_ema.reshape(config.train.train_timesteps, config.train.batch_size, *model_output_ema.shape[1:])
                sample['std_dev_t'] = std_dev_t.reshape(config.train.train_timesteps, config.train.batch_size, *std_dev_t.shape[1:])

                # grpo logic
                advantages = torch.clamp(
                    sample["advantages"][:, 0],
                    -config.train.adv_clip_max,
                    config.train.adv_clip_max,
                )
                if config.train.advantage_max is not None:
                    advantages = advantages / config.train.adv_clip_max * config.train.advantage_max

                if config.train.loss_type == 'sum_first':
                    sum_log_probs = sample['log_probs'].mean(dim=0)
                    sum_old_log_probs = sample['old_log_probs'].mean(dim=0)
                    ratio = torch.exp(sum_log_probs - sum_old_log_probs.detach())
                elif config.train.loss_type == 'exp_first':
                    log_probs = sample['log_probs'].view(-1)
                    old_log_probs = sample['old_log_probs'].view(-1)
                    ratio = torch.exp(log_probs - old_log_probs.detach())
                else:
                    raise ValueError(f"Unknown loss_type: {config.train.loss_type}")

                if accelerator.is_main_process:
                    print('config.train.clip_range', config.train.clip_range)
                    print('ratio', ratio)

                if config.train.loss_type == 'exp_first':
                    advantages = advantages.unsqueeze(0).repeat(config.train.train_timesteps, 1).view(-1)
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - config.train.clip_range,
                    1.0 + config.train.clip_range,
                )
                if config.train.loss_type == 'sum_first':
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                elif config.train.loss_type == 'exp_first':
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                else:
                    raise ValueError(f"Unknown loss_type: {config.train.loss_type}")
                if config.train.beta > 0:
                    model_output = sample['model_output'] # shape [4, 3, 16, 64, 64]
                    model_output_ref = sample['model_output_ref']
                    std_dev_t = sample['std_dev_t']
                    if config.train.kl_weight == 'ELBO':
                        kl_loss = ((model_output - model_output_ref) ** 2).mean(dim=(2,3,4), keepdim=True) / (2 * std_dev_t ** 2)
                    elif config.train.kl_weight == 'Uniform':
                        kl_loss = ((model_output - model_output_ref) ** 2).mean(dim=(2,3,4), keepdim=True)
                    else:
                        raise ValueError(f"Unknown kl_weight: {config.train.kl_weight}")
                    kl_loss = kl_loss.mean(dim=0).mean()
                    if config.train.ema_beta > 0:
                        model_output_ema = sample['model_output_ema']
                        if config.train.kl_ema_weight == 'ELBO':
                            ema_kl_loss = ((model_output - model_output_ema) ** 2).mean(dim=(2,3,4), keepdim=True) / (2 * std_dev_t ** 2)
                        elif config.train.kl_ema_weight == 'Uniform':
                            ema_kl_loss = ((model_output - model_output_ema) ** 2).mean(dim=(2,3,4), keepdim=True)
                        else:
                            raise ValueError(f"Unknown kl_weight: {config.train.kl_weight}")
                        ema_kl_loss = ema_kl_loss.mean(dim=0).mean()
                        loss = policy_loss + config.train.beta * kl_loss + config.train.ema_beta * ema_kl_loss
                    else:
                        loss = policy_loss + config.train.beta * kl_loss
                else:
                    loss = policy_loss

                info["clipfrac"].append(
                    torch.mean(
                        (
                            torch.abs(ratio - 1.0) > config.train.clip_range
                        ).float()
                    )
                )
                info["policy_loss"].append(policy_loss)
                if config.train.beta > 0:
                    info["kl_loss"].append(kl_loss)
                if config.train.ema_beta > 0:
                    info["ema_kl_loss"].append(ema_kl_loss)

                info["loss"].append(loss)

                # backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        transformer_trainable_parameters, config.train.max_grad_norm
                    )
                    log_metrics({"grad_norm": grad_norm}, step=global_step)
                optimizer.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                lr_scheduler.step()
                # log training-related stuff
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch,  "lr": lr_scheduler.get_last_lr()[0]})
                log_metrics(info, step=global_step)
                global_step += 1
                info = defaultdict(list)
        if config.train.ema:
            ema.step(transformer_trainable_parameters, global_step)
        # make sure we did an optimization step at the end of the inner epoch
        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # record the time of each stage
        log_metrics(
            {
                "training_time": training_time,
                "epoch_time": epoch_time,
            },
            step=global_step,
        )

        with torch.no_grad():
            if config.train.kl_ema_decay_type == 'constant':
                decay = config.train.kl_ema_decay
            elif config.train.kl_ema_decay_type == 'linear':
                decay = min(config.train.kl_ema_decay, 0.001 * global_step)
            if accelerator.is_main_process:
                print(f'current_step:{global_step}, decay:{decay}')
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, ema_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))


if __name__ == "__main__":
    app.run(main)

