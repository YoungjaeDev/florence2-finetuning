import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoRAConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    log_steps: int
    seed: int
    num_workers: int
    resume: bool
    resume_path: str
    prefetch_factor: int

@dataclass
class ModelConfig:
    name: str
    use_lora: bool
    trust_remote_code: bool
    dtype: str
    train_vision_encoder: bool
@dataclass
class Config:
    training: TrainingConfig
    model: ModelConfig
    lora: LoRAConfig
    distributed: dict
    paths: dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        training = TrainingConfig(**config_dict['training'])
        model = ModelConfig(**config_dict['model'])
        lora = LoRAConfig(**config_dict['lora'])
        
        return cls(
            training=training,
            model=model,
            lora=lora,
            distributed=config_dict['distributed'],
            paths=config_dict['paths']
        ) 