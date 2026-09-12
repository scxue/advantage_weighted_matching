# Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.25050-b31b1b.svg)](https://arxiv.org/abs/2509.25050)
[![PDF](https://img.shields.io/badge/PDF-download-blue.svg)](https://arxiv.org/pdf/2509.25050.pdf)

[Shuchen Xue](https://scxue.github.io/),  [Chongjian Ge](https://chongjiange.github.io/),  [Shilong Zhang](https://jshilong.github.io/),  [Yichen Li](https://people.csail.mit.edu/yichenl/),  [Zhi-Ming Ma](http://homepage.amss.ac.cn/research/homePage/8eb59241e2e74d828fb84eec0efadba5/myHomePage.html)


University of Chinese Academy of Sciences · Adobe Research · HKU · MIT  



Efficient reinforcement learning for text-to-image diffusion models. This implementation supports Stable Diffusion 3.5 with various reward functions.


## Quick Start

### Environment Setup

```bash
# Main training and Geneval reward environments are separate.
# A CUDA compiler is required for MMCV; default target is H100.
# For A100: export TORCH_CUDA_ARCH_LIST=8.0
bash setup_env.sh
```

See [installation, dependencies and portable paths](setup/README.md) before
installing. The scripts preserve existing environments unless
`AWM_UPDATE_ENV=1` is explicitly set. OCR requires the optional
`AWM_INSTALL_OCR=1` setup flag.

### Environment Variables

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
# Optional; defaults are checkout-relative and independent of the launch directory.
export AWM_OUTPUT_DIR="/your/storage/awm"
export AWM_DATA_DIR="/your/datasets"  # contains geneval/, ocr/, pickscore/
```

### Training

#### Single-Node (8 GPUs)

```bash
# Geneval on SD3.5
bash train_awm_scripts_geneval/train_single_node.sh

# PickScore on SD3.5
bash train_awm_scripts_pickscore/train_single_node.sh

# OCR on SD3.5
bash train_awm_scripts_ocr/train_single_node.sh
```

#### Multi-Node (4 nodes × 8 GPUs)

```bash
# Master node
bash train_awm_scripts_geneval/train_multi_nodes.sh # node 0

# Worker nodes
bash train_awm_scripts_geneval/train_multi_nodes.sh  # node 1
bash train_awm_scripts_geneval/train_multi_nodes.sh  # node 2
bash train_awm_scripts_geneval/train_multi_nodes.sh  # node 3
```


## Configuration

Configurations in [advantage_weighted_matching/config/dgx_awm.py](advantage_weighted_matching/config/dgx_awm.py):

### Available Configs

#### SD3.5 Configs
- `geneval_sd3_no_cfg_1node` - GenEval
- `geneval_sd3_no_cfg_2nodes` - GenEval
- `geneval_sd3_no_cfg_4nodes` - GenEval
- `ocr_sd3_no_cfg_1node` - OCR
- `ocr_sd3_no_cfg_4nodes` - OCR
- `pickscore_sd3_no_cfg_1node` - PickScore
- `pickscore_sd3_no_cfg_4nodes` - PickScore

The current task configs share `ghuber_power=0.5`,
`time_type="discrete_with_init"`, `base_kl_cfg_scale=4.5`,
`share_prompt_training_noise=True`, and `logical_ranks=32`.
Rollout/evaluation guidance remains 1.0; base-KL CFG applies only to the frozen
base reference. Training noise is shared within each prompt/epoch, not rollout
noise. Task-specific settings remain different:

| Task | Base KL beta | Images/epoch | Train batch/GPU | Logical rollout rounds | Eval interval |
| --- | --- | --- | --- | --- | --- |
| GenEval | 0.003 | 1728 | 3 | 9 | 50 epochs |
| OCR | 0.003 | 1728 | 3 | 9 | 20 epochs |
| PickScore | 0.0001 | 576 | 2 | 3 | 50 epochs |

Each node config assumes eight GPUs. Physical rollout batches and accumulation
scale with the number of logical ranks handled per GPU. Full checkpoints are
saved every 10 updates at completed epoch boundaries. See
[resume and input-layout documentation](advantage_weighted_matching/docs/resume.md)
for recovery, compatibility limits and tests. These defaults supersede the
original recipes.

### Key Parameters

```python
# Model
config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"   # Backbone diffusion model (SD3.5 Medium)
config.resolution = 512                                               # Training / sampling resolution

# Dataset & prompting
config.dataset = "dataset/{geneval|ocr|pickscore}"                    # Dataset used for training and evaluation
config.prompt_fn = "geneval | general_ocr"                            # Prompt generation function

# Sampling (data collection)
gpu_numbers = 8                                                       # * config.sample.num_batches_per_epoch = constant
config.sample.num_steps = 14                                          # Number of diffusion steps during rollout
config.sample.eval_num_steps = 40                                     # Diffusion steps used for evaluation
config.sample.guidance_scale = 1.0                                    # CFG scale in rollout (1.0 = no CFG; 3.0/4.5 for CFG)
config.sample.train_batch_size = 6                                    # Prompts per GPU per rollout batch
config.sample.test_batch_size = 6                                     # Batch size for evaluation sampling
config.sample.num_image_per_prompt = 24                               # Images sampled per prompt (group size)
config.sample.num_batches_per_epoch = = 24 * 24 // gpu_numbers // 2   # Rollout batches per epoch
config.sample.noise_level = 0.4                                       # Noise level during sampling (ODE=0.0; SDE > 0.0)
config.sample.sde_frac = 0.0                                          # Fraction of SDE steps in DDIM(0 = deterministic ODE)
                                                                      # 0.5: first 50% stochastic DDIM, remaining 50% DDIM
config.sample.global_std = True                                       # Use global std for advantage normalization
config.train.use_sa_solver = True                                     # Use SA-Solver for rollout; if False: DDIM sampling

# Training (gradient updates)
config.train.batch_size = 1                                           # for current imple, actual micro-bs is config.train.     
                                                                      # batch_size * config.train.train_timesteps
config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch                                                 # Set to maintain on policy
config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch                                                  
config.train.train_timesteps = 6                                      # Number of diffusion timesteps trained
config.train.learning_rate = 3e-4                                     # AdamW learning rate

# Advantage / KL control
config.train.beta = 0.003                                             # KL regularization coefficient
config.train.base_kl_cfg_scale = 4.5                                  # Base reference only; PickScore beta remains 0.0001
config.train.clip_range = 1                                           # Advantage clipping epsilon; currently not work (on policy)
config.train.advantage_max = 1                                        # Max advantage abs value
config.train.kl_weight = "Uniform"                                    # Time weighting for KL term; Uniform or ELBO

# Loss shaping
config.train.loss_type = "exp_first"                                  # Loss formulation; exp_first or sum_first; gradient is equivalent for on-policy
config.train.weighting = "ghuber"                                     # Robust loss weighting scheme
config.train.ghuber_power = 0.5                                       # Power for generalized Huber weighting
config.train.timestep_fraction = 0.9                                  # Fraction of timesteps used for loss

# KL-EMA (mimic TRPO)
config.train.kl_ema_weight = "Uniform"                                # Time weighting for KL-EMA
config.train.kl_ema_decay = 0.3                                       # EMA decay rate
config.train.kl_ema_decay_type = "linear"                             # EMA decay schedule
config.train.kl_ema_decay_rate = 0.001                                # min(rate * global_step, kl_ema_decay)
config.train.ema_beta = 1                                             # Scaling factor for EMA KL

# EMA & optimization tricks
config.train.ema = True                                               # Enable EMA for model weights
config.train.ema_decay = 0.99                                         # EMA decay rate
config.train.ema_update_step_interval = 1                             # EMA update frequency

# Time sampling
config.time_type = "discrete_with_init"                               # Initial schedule index plus 5 stratified indices
config.train.share_prompt_training_noise = True                      # Shared per prompt/epoch, independent of physical rank

# Rewards
config.reward_fn = {                                                   # Reward function(s) and weights
    "geneval": 1.0,        # or "ocr": 1.0 / "pickscore": 1.0
}

# Logging & bookkeeping
config.per_prompt_stat_tracking = True                                # Track per-prompt statistics
config.num_epochs = 100000                                            # Total training epochs
config.checkpoint_every_steps = 10                                    # Full epoch-boundary checkpoints
config.logical_ranks = 32                                            # Fixed input/RNG layout
config.logical_rounds = 9                                            # Geneval/OCR: 9; PickScore: 3
config.eval_freq = 20–50                                              # Evaluation frequency (epochs)
```

### Hyperparameter Guidelines

**CFG Scale during Rollout and Evaluation; KL Reference Model**

In our early experiment, we used a guidance scale of $\text{cfg}=4.5$ for both rollout and evaluation and set the reference model to the base model. 

In later experiments, we observe that a guidance scale of $\text{cfg}=1.0$ for both rollout and evaluation also works (similar to DanceGRPO, FlowGRPO-fast and DiffusionNFT), which saves about half the computation during rollout. We also observe that setting a large coefficient for KL with respect to the EMA model (using a linear warmup EMA rate min(0.001* iter, 0.3)), and a small coefficient for KL with respect to the base model will be better.

Some of my personal views on CFG:

(1) From the theoretical perspective of the consistency between rollout policy and train policy, as CFG should not be used in diffusion/flow loss, the rollout policy should be $\text{cfg}=1.0$. However, in practice, RL using rollout $\text{cfg}=2.0,3.0,4.0$ also works.

(2) Rollout with $\text{cfg}=1.0$ enables the model to quickly gain the ability to generate without cfg in 10 steps. I attribute this fact to both RL and CFG sharpening the distribution.

(3) My experience: the evaluation cfg should be set equal to or slightly larger than the cfg during rollout. For example, when training the model with rollout $\text{cfg}=1.0$, using a cfg scale of 1.5 for inference will generally be visually better, at roughly twice the computational cost.

(4) There is one thing I have not done a systematic ablation on: as the base model does not have the ability to generate without cfg, the KL term with respect to the base model might not be perfect. My current solution is using a large coefficient for KL with respect to the EMA model, and a small coefficient for KL with respect to the base model, as the EMA model quickly follows the running model to learn the ability to generate without cfg. Using a $\text{cfg}=2.0$ base model in KL divergence might be better.

**Training Batch Relationship**:
```python
config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch                                                 # Set to maintain on policy
config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
```


## Acknowledgement

Our code is based on [DDPO-Pytorch](https://github.com/kvablack/ddpo-pytorch), [Flow-GRPO](https://github.com/yifan123/flow_grpo)

## Citation

If you find this repo helpful, please consider citing:

```bibtex
@article{xue2025advantage,
  title={Advantage weighted matching: Aligning rl with pretraining in diffusion models},
  author={Xue, Shuchen and Ge, Chongjian and Zhang, Shilong and Li, Yichen and Ma, Zhi-Ming},
  journal={arXiv preprint arXiv:2509.25050},
  year={2025}
}
```
