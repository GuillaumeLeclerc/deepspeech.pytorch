from numpy.random import shuffle
from types import SimpleNamespace
import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.loader.data_module import TestDataset
from deepspeech_pytorch.validation import run_evaluation
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader, IterableDataset


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")
    print(device)

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    test_dataset = TestDataset(
        cfg.test_path,
        labels=model.labels,
        batch_size=cfg.batch_size,
        shuffle_size=0,
        audio_conf=model.spect_cfg,
        normalize=True,
        testing=True,
        augmentation_conf=SimpleNamespace(**{'kenansville_strength': 500, 'noise_factor': 1})
        )

    wer, cer = run_evaluation(
        test_loader=test_dataset,
        device=device,
        model=model,
        decoder=decoder,
        target_decoder=target_decoder,
        precision=cfg.model.precision
    )

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
