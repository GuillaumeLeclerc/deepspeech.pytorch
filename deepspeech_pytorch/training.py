import json

from omegaconf import OmegaConf
import hydra
from deepspeech_pytorch.checkpoint import GCSCheckpointHandler, FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, GCSCheckpointConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech
from hydra.utils import to_absolute_path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if cfg.trainer.checkpoint_callback:
        if OmegaConf.get_type(cfg.checkpoint) is GCSCheckpointConfig:
            checkpoint_callback = GCSCheckpointHandler(
                cfg=cfg.checkpoint
            )
        else:
            checkpoint_callback = FileCheckpointHandler(
                cfg=cfg.checkpoint
            )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=cfg.trainer.gpus > 1
    )

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )
    logger = WandbLogger(project='audio-prod')

    params = OmegaConf.to_container(cfg.trainer)
    del params['_target_']
    del params['callbacks']
    del params['replace_sampler_ddp']
    del params['logger']


    trainer = Trainer(
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback] if cfg.trainer.checkpoint_callback else None,
        logger=logger,
        **params
    )

    trainer.fit(model, data_loader)
