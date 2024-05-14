from dataclasses import dataclass, asdict
from typing import Any, Optional, cast, Union


@dataclass
class SAEConfig:
    
    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-4l"
    hook_point: str = "blocks.{layer}.hook_resid_pre"
    hook_point_layer: int = 2
    dataset_path = "NeelNanda/c4-code-tokenized-2b"
    is_dataset_tokenized=True,
    context_size: int = 1024

    # SAE Parameters
    d_in: int = 512
    expansion_factor: int = 8
    d_sae: Optional[int] = None

    # Activation Store Parameters
    n_batches_in_buffer: int = 100
    store_batch_size: int = 8

    # Sparse Attribution Parameters
    attrib_sparsity_coeff: float = 1.5e-4
    unexplained_attrib_coeff: float = 0.5
    mse_coefficient: float = 0.5

    # Training Parameters
    l1_coefficient: float = 1.5e-4
    lp_norm: float = 1
    lr: float = 3e-4
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup
    )
    l1_warm_up_steps: Optional[int] = None
    lr_warm_up_steps: int = 500
    train_batch_size: int = 4096
    n_training_tokens: int = int(5e8) #int(1e9)

    dead_feature_threshold: float = 2e7
    steps_between_resample: int = 25000 # Anthropic does every 25000
    tune_resample: bool = False

    checkpoint_frequency: Optional[int] = 10000

    # WANDB
    log_to_wandb: bool = True
    wandb_project: Optional[str] = "sparse_attribution"
    run_name: Optional[str] = None
    wandb_log_frequency: int = 100

    # Misc
    device: str = "cpu"

    def __post_init__(self):

        if self.log_to_wandb and self.wandb_project is None:
            raise ValueError("If log_to_wandb is True, wandb_project must be set.")

        if self.d_sae is None:
            self.d_sae = self.d_in * self.expansion_factor

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}"

        print(f"Run name: {self.run_name}")
    
    def to_dict(self):
        return asdict(self)
