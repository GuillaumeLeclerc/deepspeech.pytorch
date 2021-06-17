import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader, IterableDataset

from glob import glob
import json
from types import SimpleNamespace

from numpy.lib.function_base import hamming
from deepspeech_pytorch.enums import SpectrogramWindow, RNNType
import torch
import librosa
import numpy as np
import torchaudio
import webdataset as wds
import pytorch_lightning as pl

torchaudio.set_audio_backend("sox_io")

from deepspeech_pytorch.configs.train_config import SpectConfig, AugmentationConfig
from deepspeech_pytorch.loader.spec_augment import spec_augment

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, input_percentages, target_sizes


class DataSet(IterableDataset):

    def __init__(self,
                 data_path,
                 labels,
                 batch_size,
                 shuffle_size,
                 audio_conf: SpectConfig,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None,
                 testing=False):
        self.kenansville_strength = augmentation_conf.kenansville_strength
        self.noise_factor = augmentation_conf.noise_factor
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        self.testing = testing
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

        all_files = glob(data_path)
        dataset = wds.WebDataset(all_files, shardshuffle=(shuffle_size > 0))
        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.decode(
                wds.torch_audio)
        dataset = dataset.to_tuple("flac", "txt")
        dataset = dataset.map_tuple(self.load_audio, self.load_label)
        dataset = dataset.batched(batch_size, _collate_fn)
        self.dataset = dataset
        
    def __iter__(self):
        return iter(self.dataset)

    def kenansville_attack(self, x):
        threshold = 10**(-self.kenansville_strength / 10)
        x_fft = np.fft.fft(x)
        x_psd = np.abs(x_fft) ** 2
        threshold *= np.sum(x_psd)
        x_psd_ind = np.argsort(x_psd)
        dc_ind = np.where(x_psd_ind == 0)[0][0]
        reordered = x_psd[x_psd_ind]
        id = np.searchsorted(np.cumsum(reordered), threshold)
        id -= id % 2
        if (dc_ind < id) ^ (len(x) % 2 == 0 and len(x) / 2 < id):
            id -= 1
        x_fft[x_psd_ind[:id]] = 0
        x_ifft = np.fft.ifft(x_fft)
        return np.real(x_ifft).astype(np.float32)

    def noise_attack(self, signal):
        threshold = np.sqrt(10**(-20 / 10))
        threshold *= self.noise_factor
        perturbation = np.random.randn(*signal.shape)
        perturbation /= np.linalg.norm(perturbation)
        perturbation *= np.linalg.norm(signal) * threshold
        new_signal = perturbation + signal
        return new_signal
    
    def attack(self, signal):
        signal = signal.numpy()
        if self.testing:
            return signal

        if self.noise_factor == 0:
            return self.kenansville_attack(signal)
            
        if np.random.randn() > 0:
            return self.kenansville_attack(signal)
        else:
            return self.noise_attack(signal)

    def load_audio(self, data):
        tensor, sample_rate = data
        assert sample_rate == self.sample_rate
        y = self.attack(tensor[0])

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        try:
            if self.aug_conf and self.aug_conf.spec_augment:
                spect = spec_augment(spect)
        except:
            pass

        return spect
    
    def load_label(self, label):
        return [self.labels_map.get(x) for x in list(label)]
    
class TestDataset(DataSet):
    def __init__(self,
                 data_path,
                 labels,
                 batch_size,
                 shuffle_size,
                 audio_conf: SpectConfig,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None,
                 testing=False):
        self.kenansville_strength = augmentation_conf.kenansville_strength
        self.noise_factor = augmentation_conf.noise_factor
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        self.testing = testing
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

        all_files = glob(data_path)
        dataset = wds.WebDataset(all_files, shardshuffle=(shuffle_size > 0))
        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.decode(
                wds.torch_audio)
        dataset = dataset.to_tuple("flac", "txt")
        dataset = dataset.map_tuple(self.load_audio, self.load_label)
        dataset = dataset.map(lambda x: _collate_fn(zip(x[0], x[1])))
        self.dataset = dataset

    def attack(self, signal):
        signal = signal.numpy()
        return self.noise_attack(signal)
    
    def load_label(self, label):
        prev = super().load_label(label)
        return [prev for _ in range(9)]

    def load_audio(self, data):
        tensor, sample_rate = data
        assert sample_rate == self.sample_rate
        results = []
        for i in range(9):
            y = self.attack(tensor[0])

            n_fft = int(self.sample_rate * self.window_size)
            win_length = n_fft
            hop_length = int(self.sample_rate * self.window_stride)
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            spect, phase = librosa.magphase(D)
            spect = np.log1p(spect)
            spect = torch.FloatTensor(spect)
            if self.normalize:
                mean = spect.mean()
                std = spect.std()
                spect.add_(-mean)
                spect.div_(std)

            try:
                if self.aug_conf and self.aug_conf.spec_augment:
                    spect = spec_augment(spect)
            except:
                pass

            results.append(spect)
        return torch.stack(results)



class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 labels: list,
                 data_cfg: DataConfig,
                 normalize: bool,
                 is_distributed: bool):
        super().__init__()
        self.train_path = to_absolute_path(data_cfg.train_path)
        self.val_path = to_absolute_path(data_cfg.val_path)
        self.labels = labels
        self.data_cfg = data_cfg
        self.spect_cfg = data_cfg.spect
        self.aug_cfg = data_cfg.augmentation
        self.normalize = normalize
        self.is_distributed = is_distributed

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_path)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=None,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_path)
        val_loader = DataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=None,
            pin_memory=True
        )
        return val_loader

    def _create_dataset(self, input_path):
        return DataSet(input_path, self.labels, self.data_cfg.batch_size,
                       self.data_cfg.shuffle_size,
                       self.spect_cfg, self.normalize, self.aug_cfg)
