from typing import Dict, List
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class BaseTrainingConfig:
    # Dir
    logging_dir: str
    output_dir: str

    # Logger and checkpoint
    logger: str = 'tensorboard'
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 20
    valid_epochs: int = 100
    valid_steps: int = 0
    save_model_epochs: int = 100
    resume_from_checkpoint: str = None

    # Training
    seed: int = None
    num_epochs: int = 200
    train_batch_size: int = 1
    dataloader_num_workers: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = None
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False

    # Dataset
    train_dataset_name: str = None
    train_dataset_config_name: str = None
    train_data_dir: str = None
    train_data_files: List[str] = None
    train_metadata_path: str = None
    valid_dataset_name: str = None
    valid_dataset_config_name: str = None
    valid_data_dir: str = None
    valid_data_files: List[str] = None
    valid_split_name: str = 'test'
    cache_dir: str = None
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False
    downsample_mode: str = 'bilinear'
    is_imagenet: bool = False

    # Hub
    push_to_hub: bool = False
    hub_model_id: str = ''


__TYPE_CLS_DICT = {}


def register_training_config_func(training_config_type: str, training_config_cls):
    __TYPE_CLS_DICT[training_config_type] = training_config_cls


def register_training_config(training_config_type: str):
    def wrapper(training_config_cls):
        return register_training_config_func(training_config_type, training_config_cls)
    return wrapper


def load_training_config(config_path: str) -> Dict[str, BaseTrainingConfig]:
    data_dict = OmegaConf.load(config_path)

    # The config must have a "base" key
    base_cfg_dict = data_dict.pop('base')

    # The config must have one another model config
    assert len(data_dict) == 1
    model_key = next(iter(data_dict))
    model_cfg_dict = data_dict[model_key]
    model_cfg_cls = __TYPE_CLS_DICT[model_key]

    return {'base': BaseTrainingConfig(**base_cfg_dict),
            model_key: model_cfg_cls(**model_cfg_dict)}
