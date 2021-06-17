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


class DataSet:

    def __init__(self,
                 labels,
                 audio_conf: SpectConfig,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None):
        self.kenansville_strength = augmentation_conf.kenansville_strength
        self.noise_factor = augmentation_conf.noise_factor
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

        
        

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

        if self.aug_conf and self.aug_conf.spec_augment:
            spect = spec_augment(spect)

        return spect
    
    def load_label(self, label):
        return [self.labels_map.get(x) for x in list(label)]


if __name__ == '__main__':
    with open('./labels.json') as label_file:
        labels = json.load(label_file)

    audio_conf = SimpleNamespace(**{
        'window_stride': 0.1,
        'window_size': 0.2,
        'sample_rate': 16000,
        'window': SpectrogramWindow.hamming
    })
    aug_conf = SimpleNamespace(**{
        'kenansville_strength': 20,
        'noise_factor': 5,
        'spec_augment': False
    })
    processor = DataSet(labels, audio_conf, True, aug_conf)

    split = 'train-clean-360'
    pattern = f'/mnt/nfs/datasets/librispeech/webdataset/'
    all_files = glob(pattern + f'*{split}*')
    dataset = wds.WebDataset(all_files)
    dataset = dataset.decode(
            wds.torch_audio)
    dataset = dataset.to_tuple("flac", "txt")
    dataset = dataset.map_tuple(processor.load_audio, processor.load_label)
    dataset = dataset.batched(4, _collate_fn)
    
    for i in dataset:
        print(len(i), i)
        break
