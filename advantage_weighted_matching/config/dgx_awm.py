import ml_collections
import imp
import os
import datetime
from pathlib import Path

base_awm = imp.load_source("base_awm", os.path.join(os.path.dirname(__file__), "base_awm.py"))

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def _output_root():
    return Path(os.environ.get("AWM_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()


def _dataset_path(task):
    root = Path(os.environ.get("AWM_DATA_DIR", PACKAGE_ROOT / "dataset")).expanduser().resolve()
    return str(root / task)


def _save_path(task, nodes, timestamp):
    label = "1node" if nodes == 1 else f"{nodes}nodes"
    return str(_output_root() / task / label / timestamp)


def compressibility():
    config = base_awm.get_config()
    config.logdir = str(_output_root() / "logs")

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = _dataset_path("pickscore")

    config.num_epochs = 100
    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def _geneval_sd3_no_cfg(nodes):
    """Shared Geneval recipe; each node is assumed to have eight GPUs."""
    config = compressibility()
    config.logical_ranks = 32
    config.train.share_prompt_training_noise = True
    config.dataset = _dataset_path("geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0

    gpu_numbers = nodes * 8
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 24 * 24 // gpu_numbers // 2 # 24 gpu uses 24 num_batches_per_epoch; for on policy, we additionally // 2
    config.sample.test_batch_size = 6 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.noise_level = 0.4
    config.sample.sde_frac = 0.0

    config.train.batch_size = 3 # train.batch_size must divide sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.003
    config.train.base_kl_cfg_scale = 4.5
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema=True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.5
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.kl_ema_decay_rate = 0.001
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.sample.global_std=True

    config.num_epochs = 100000
    config.save_freq = 50 # epoch
    config.eval_freq = 50
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }

    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    config.train.clip_range = 1
    config.lora_type = "attn"
    config.time_type = "discrete_with_init"
    config.time_shift = 3.0
    config.train.decay_steps = 10000000000
    config.train.ema_decay=0.99
    config.train.ema_update_step_interval=1



    node_label = "1node" if nodes == 1 else f"{nodes}nodes"
    config.run_name = (
        f"geneval_sd3_{node_label}_shared_prompt_noise_beta2=0.999_kl_beta=0.003"
        "_kl_ema_rate=0.001_ghuber=0.5_with_init_base_cfg=4.5"
    )
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.save_dir = _save_path("geneval", nodes, unique_id)
    return config

def geneval_sd3_no_cfg_1node():
    return _geneval_sd3_no_cfg(1)


def geneval_sd3_no_cfg_2nodes():
    return _geneval_sd3_no_cfg(2)


def geneval_sd3_no_cfg_4nodes():
    return _geneval_sd3_no_cfg(4)


def ocr_sd3_no_cfg_1node():
    config = compressibility()
    config.logical_ranks = 32
    config.train.share_prompt_training_noise = True
    config.train.base_kl_cfg_scale = 4.5
    config.dataset = _dataset_path("ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0

    # n*A800
    gpu_numbers = 8
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 24 * 24 // gpu_numbers // 2 # 24 gpu uses 24 num_batches_per_epoch; for on policy, we additionally // 2
    config.sample.test_batch_size = 6 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.noise_level = 0.4
    config.sample.sde_frac = 0.0

    config.train.batch_size = 3 # train.batch_size must divide sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.003
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema=True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.5
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.sample.global_std=True

    config.num_epochs = 100000
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.reward_fn = {
        # "geneval": 1.0,
        "ocr": 1.0
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    config.train.clip_range = 1
    config.lora_type = "attn"
    config.time_type = "discrete_with_init"
    config.time_shift = 3.0
    config.train.decay_steps = 10000000000
    config.train.ema_decay=0.99
    config.train.ema_update_step_interval=1

    config.run_name = f"ocr_sd3_1node"
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.save_dir = _save_path("ocr", 1, unique_id)
    return config

def ocr_sd3_no_cfg_4nodes():
    config = compressibility()
    config.logical_ranks = 32
    config.train.share_prompt_training_noise = True
    config.train.base_kl_cfg_scale = 4.5
    config.dataset = _dataset_path("ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0

    # n*A800
    gpu_numbers = 32
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 24 * 24 // gpu_numbers // 2 # 24 gpu uses 24 num_batches_per_epoch; for on policy, we additionally // 2
    config.sample.test_batch_size = 6 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.noise_level = 0.4
    config.sample.sde_frac = 0.0

    config.train.batch_size = 3 # train.batch_size must divide sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.003
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema=True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.5
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.sample.global_std=True

    config.num_epochs = 100000
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.reward_fn = {
        # "geneval": 1.0,
        "ocr": 1.0
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    config.train.clip_range = 1
    config.lora_type = "attn"
    config.time_type = "discrete_with_init"
    config.time_shift = 3.0
    config.train.decay_steps = 10000000000
    config.train.ema_decay=0.99
    config.train.ema_update_step_interval=1

    config.run_name = f"ocr_sd3_4nodes"
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.save_dir = _save_path("ocr", 4, unique_id)
    return config


def pickscore_sd3_no_cfg_1node():
    config = compressibility()
    config.logical_ranks = 32
    config.train.share_prompt_training_noise = True
    config.train.base_kl_cfg_scale = 4.5
    config.logical_rounds = 3
    config.dataset = _dataset_path("pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0

    # n*A800
    gpu_numbers = 8
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 8 * 24 // gpu_numbers // 2 # 24 gpu uses 24 num_batches_per_epoch; for on policy, we additionally // 2
    config.sample.test_batch_size = 6 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.noise_level = 0.0
    config.sample.sde_frac = 0.0

    config.train.batch_size = 2 # train.batch_size must divide sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.0001
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema=True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.5
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.sample.global_std=True

    config.num_epochs = 100000
    config.save_freq = 50 # epoch
    config.eval_freq = 50
    config.reward_fn = {
        # "geneval": 1.0,
        "pickscore": 1.0
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    config.train.clip_range = 1
    config.lora_type = "attn"
    config.time_type = "discrete_with_init"
    config.time_shift = 3.0
    config.train.decay_steps = 10000000000
    config.train.ema_decay=0.99
    config.train.ema_update_step_interval=1


    config.run_name = f"pickscore_sd3_1node"
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.save_dir = _save_path("pickscore", 1, unique_id)
    return config

def pickscore_sd3_no_cfg_4nodes():
    config = compressibility()
    config.logical_ranks = 32
    config.train.share_prompt_training_noise = True
    config.train.base_kl_cfg_scale = 4.5
    config.logical_rounds = 3
    config.dataset = _dataset_path("pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0

    # n*A800
    gpu_numbers = 32
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 8 * 24 // gpu_numbers // 2 # 24 gpu uses 24 num_batches_per_epoch; for on policy, we additionally // 2
    config.sample.test_batch_size = 6 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.noise_level = 0.0
    config.sample.sde_frac = 0.0

    config.train.batch_size = 2 # train.batch_size must divide sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.0001
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema=True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.5
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.sample.global_std=True

    config.num_epochs = 100000
    config.save_freq = 50 # epoch
    config.eval_freq = 50
    config.reward_fn = {
        # "geneval": 1.0,
        "pickscore": 1.0
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    config.train.clip_range = 1
    config.lora_type = "attn"
    config.time_type = "discrete_with_init"
    config.time_shift = 3.0
    config.train.decay_steps = 10000000000
    config.train.ema_decay=0.99
    config.train.ema_update_step_interval=1

    config.run_name = f"pickscore_sd3_4nodes"
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.save_dir = _save_path("pickscore", 4, unique_id)
    return config

def get_config(name):
    return globals()[name]()
