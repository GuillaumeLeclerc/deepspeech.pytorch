from dataclasses import dataclass, field
from typing import Any, List, Tuple
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from omegaconf import MISSING

from hydra_configs.pytorch_lightning.callbacks import ModelCheckpointConf
from omegaconf import MISSING

@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    checkpoint_callback: bool = True
    callbacks: Any = None  # Optional[List[Callback]]
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Any = None  # Union[int, str, List[int], NoneType]
    auto_select_gpus: bool = False
    tpu_cores: Any = None  # Union[int, str, List[int], NoneType]
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]]
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    flush_logs_every_n_steps: int = 100
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    sync_batchnorm: bool = False
    precision: int = 32
    weights_summary: Optional[str] = "top"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    profiler: Any = None  # Union[BaseProfiler, bool, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Any = False  # Union[str, bool]
    prepare_data_per_node: bool = True
    plugins: Any = None   # Union[str, list, NoneType]
    amp_backend: str = "native"
    amp_level: str = "O2"
    distributed_backend: Optional[str] = None
    move_metrics_to_cpu: bool = False

from deepspeech_pytorch.enums import SpectrogramWindow, RNNType

defaults = [
    {"optim": "adam"},
    {"model": "bidirectional"},
    {"checkpoint": "file"}
]


@dataclass
class SpectConfig:
    sample_rate: int = 16000  # The sample rate for the data/model features
    window_size: float = .02  # Window size for spectrogram generation (seconds)
    window_stride: float = .01  # Window stride for spectrogram generation (seconds)
    window: SpectrogramWindow = SpectrogramWindow.hamming  # Window type for spectrogram generation


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False  # Use random tempo and gain perturbations.
    spec_augment: bool = False  # Use simple spectral augmentation on mel spectograms.
    noise_dir: str = ''  # Directory to inject noise into audio. If default, noise Inject not added
    noise_prob: float = 0.4  # Probability of noise being added per sample
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.5  # Maximum noise levels to sample from. Maximum 1.0
    kenansville_strength: float = 20.0
    noise_factor:float = 5.0


@dataclass
class DataConfig:
    train_path: str = 'data/train_manifest.csv'
    val_path: str = 'data/val_manifest.csv'
    batch_size: int = 64  # Batch size for training
    shuffle_size: int = 512  # Size of the shuffling pool
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output
    spect: SpectConfig = SpectConfig()
    augmentation: AugmentationConfig = AugmentationConfig()


@dataclass
class BiDirectionalConfig:
    rnn_type: RNNType = RNNType.lstm  # Type of RNN to use in model
    hidden_size: int = 1024  # Hidden size of RNN Layer
    hidden_layers: int = 5  # Number of RNN layers


@dataclass
class UniDirectionalConfig(BiDirectionalConfig):
    lookahead_context: int = 20  # The lookahead context for convolution after RNN layers


@dataclass
class OptimConfig:
    learning_rate: float = 1.5e-4  # Initial Learning Rate
    learning_anneal: float = 0.99  # Annealing applied to learning rate after each epoch
    weight_decay: float = 1e-5  # Initial Weight Decay


@dataclass
class SGDConfig(OptimConfig):
    momentum: float = 0.9


@dataclass
class AdamConfig(OptimConfig):
    eps: float = 1e-8  # Adam eps
    betas: tuple = (0.9, 0.999)  # Adam betas


@dataclass
class GCSCheckpointConfig(ModelCheckpointConf):
    gcs_bucket: str = MISSING  # Bucket to store model checkpoints e.g bucket-name
    gcs_save_folder: str = MISSING  # Folder to store model checkpoints in bucket e.g models/


@dataclass
class DeepSpeechTrainerConf(TrainerConf):
    callbacks: Any = MISSING


@dataclass
class DeepSpeechConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    optim: Any = MISSING
    model: Any = MISSING
    checkpoint: ModelCheckpointConf = MISSING
    trainer: DeepSpeechTrainerConf = DeepSpeechTrainerConf()
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    seed: int = 123456  # Seed for generators
    load_auto_checkpoint: bool = False  # Automatically load the latest checkpoint from save folder
