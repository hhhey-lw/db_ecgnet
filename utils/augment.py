import numpy as np
import torch
from torch import Tensor


# reference paper:  
#   1. Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction: A Multi-Dataset Study
#   2. Practical intelligent diagnostic algorithm for wearable 12-lead ECG via self-supervised learning on large-scale dataset
#   ⭐ May be, Simple augmentation can also lead to dramatic changes in the semantic information of the ECG signal

def ECGAugment(x: Tensor) -> Tensor:
    """
        x: batch, channel, length
    """
    aug_list = [1, 2, 3]
    aug_que = np.unique(np.random.choice(aug_list, 1))
    np.random.shuffle(aug_que)
    for aug_type in aug_que:
        if aug_type == 0:
            x = signal_reversal(x)
        elif aug_type == 1:
            x = signal_dropout(x)
        elif aug_type == 2:
            x = gaussian_noise(x)
        elif aug_type == 3:
            x = channel_permutation(x)
    return x


def signal_reversal(x: Tensor) -> Tensor:
    return x.flip(dims=[-1])


def channel_permutation(x: Tensor) -> Tensor:
    indices = torch.randperm(x.shape[1])
    return x[:, indices, :]


def gaussian_noise(x: Tensor) -> Tensor:
    noise = torch.randn_like(x) * 0.1
    return x + noise


def signal_dropout(x: Tensor) -> Tensor:
    batch, channel, length = x.shape
    for idx in range(batch):
        window_length = np.random.randint(length // 8)
        discard_start = np.random.randint(0, length - window_length)
        x[idx, :, discard_start:discard_start + window_length] = 0
    return x


if __name__ == '__main__':
    x = torch.randn((4, 12, 1000))
    x = signal_dropout(x)
    print(x.shape)