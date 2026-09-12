# Shared prompt training randomness

Training strategy change, not a numerical-precision fix.

## Configuration and semantics

- `train.share_prompt_training_noise=True` enables sharing. Base default is False;
  All public Geneval/OCR/PickScore task factories enable it.
- Current with-init setup retains 6 times: index0 plus 5 stratified discrete
  indices. All images of the same prompt in an epoch share those 6 times.
- Each time slot has its own Gaussian tensor; the corresponding tensor is
  shared by every image in that prompt group. Different images still have
  different clean latents.
- Group identity is the tokenizer-decoded prompt text used by the advantage
  tracker, not physical rank, rollout batch, microbatch, or dataset row index.
  Repeated prompt text across rollout rounds shares the same training draws.
- A SHA256-derived 63-bit seed incorporates a version tag, experiment seed,
  epoch, and prompt text. Each microbatch regenerates its inputs using private
  torch generators. No global GPU noise cache or broadcast is required.
- Distinct prompts/epochs get fresh random draws. Discrete timestep selections
  may coincide by chance, as expected; sharing does not force different groups
  to choose disjoint times.
- Supports the existing discrete, discrete_with_init and discrete_wo_init
  modes. Continuous modes fail clearly if sharing is enabled.

## Rollout, resume, and naming

Rollout/eval pipeline source is unchanged. The trainer retains its old timestep
and noise draws (discarding their values in shared mode) before using the private
prompt-keyed generators. This preserves ambient/logical RNG consumption and the
subsequent rollout/shuffle stream. This costs extra random draws but no extra
model forwards. It does not promise equal generated images after policies differ.

The new module is included in checkpoint source signatures, and the flag is
part of the config signature. New same-setting checkpoints remain resumable
without saving a noise cache. Old independent-noise checkpoints are not silently
accepted as exact resumes under the changed code/strategy; use their original
code for legacy resume.

Run names automatically include `shared_prompt_noise` when enabled.
The sharing mechanism itself does not change KL coefficients, optimizer,
precision, rollout settings or batch sizes. Current task defaults additionally
use ghuber=0.5, with-init and base-KL CFG=4.5; see the configuration table in README.

## Tests

From the package directory, after activating the main training environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p 'test_prompt_noise.py' -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p 'test_task_layout.py' -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p 'test_resume.py' -v
```

Coverage includes the actual trainer input-preparation loop and prompt-key
attachment/shuffle/rebatch code (AST-extracted to avoid loading full SD3),
cross-microbatch sharing, time-major flattening, independent draws across prompts,
epochs and time slots, stateless regeneration, and unchanged ambient RNG.
Task-layout tests compare all 1728 Geneval/OCR or 576 PickScore sample IDs under
8/16/32-process layouts. CUDA-specific tests skip when CUDA is unavailable.

## Compatibility

Changing from independent to shared training inputs is a new experiment.
Old checkpoint source/config signatures are not silently bypassed; use the
original code for an old run, or explicitly start a new experiment with new
output paths and W&B run IDs.
