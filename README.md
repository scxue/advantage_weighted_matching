# Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.25050-b31b1b.svg)](https://arxiv.org/abs/2509.25050)
[![PDF](https://img.shields.io/badge/PDF-download-blue.svg)](https://arxiv.org/pdf/2509.25050.pdf)

[Shuchen Xue](https://scxue.github.io/),  [Chongjian Ge](https://chongjiange.github.io/),  [Shilong Zhang](https://jshilong.github.io/),  [Yichen Li](https://people.csail.mit.edu/yichenl/),  [Zhi-Ming Ma](http://homepage.amss.ac.cn/research/homePage/8eb59241e2e74d828fb84eec0efadba5/myHomePage.html)


University of Chinese Academy of Sciences · Adobe Research · HKU · MIT  



Efficient reinforcement learning for text-to-image diffusion models. This implementation supports Stable Diffusion 3.5 with various reward functions.


## Quick Start

### Environment Setup

```bash
# Automated environment setup.
# The script `setup/02_setup_env_reward_server.sh` installs MMCV and MMDetection.
# For GPUs with Ampere architecture (e.g., A100, RTX 30xx),
# please follow the comments in the script to adjust the MMCV/MMDetection installation accordingly.
bash setup_env.sh
```

### Environment Variables

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
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
- `geneval_sd3_no_cfg_4nodes` - GenEval
- `ocr_sd3_no_cfg_1node` - OCR
- `ocr_sd3_no_cfg_4nodes` - OCR
- `pickscore_sd3_no_cfg_1node` - PickScore
- `pickscore_sd3_no_cfg_4nodes` - PickScore

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
config.train.clip_range = 1                                           # Advantage clipping epsilon; currently not work (on policy)
config.train.advantage_max = 1                                        # Max advantage abs value
config.train.kl_weight = "Uniform"                                    # Time weighting for KL term; Uniform or ELBO

# Loss shaping
config.train.loss_type = "exp_first"                                  # Loss formulation; exp_first or sum_first; gradient is equivalent for on-policy
config.train.weighting = "ghuber"                                     # Robust loss weighting scheme
config.train.ghuber_power = 0.25                                      # Power for generalized Huber weighting
config.train.timestep_fraction = 0.9                                  # Fraction of timesteps used for loss

# KL-EMA (mimic TRPO)
config.train.kl_ema_weight = "Uniform"                                # Time weighting for KL-EMA
config.train.kl_ema_decay = 0.3                                       # EMA decay rate
config.train.kl_ema_decay_type = "linear"                             # EMA decay schedule
config.train.ema_beta = 1                                             # Scaling factor for EMA KL

# EMA & optimization tricks
config.train.ema = True                                               # Enable EMA for model weights
config.train.ema_decay = 0.99                                         # EMA decay rate
config.train.ema_update_step_interval = 1                             # EMA update frequency

# Time sampling
config.time_type = "discrete_wo_init"                                 # Time sampling strategy

# Rewards
config.reward_fn = {                                                   # Reward function(s) and weights
    "geneval": 1.0,        # or "ocr": 1.0 / "pickscore": 1.0
}

# Logging & bookkeeping
config.per_prompt_stat_tracking = True                                # Track per-prompt statistics
config.num_epochs = 100000                                            # Total training epochs
config.save_freq = 20–50                                              # Checkpoint saving frequency (epochs)
config.eval_freq = 20–50                                              # Evaluation frequency (epochs)
```

### Hyperparameter Guidelines

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
