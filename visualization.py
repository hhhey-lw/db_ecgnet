import os

import numpy as np
import torch
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
from biosppy.signals import ecg as ECGTool
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader

from data.dataset import load_datasets
from models.db_ecg_net import DB_ECGNet

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = './checkpoints/CPSC_DBECGNet_rhythm_checkpoint.pth'

# download link: https://drive.google.com/file/d/1juLu0u95YvxKsNDddltx5qrJjyva6tul/view?usp=sharing
test_data_fn = 'checkpoints/cpsc_test_data.pth'
# download link: https://drive.google.com/file/d/1ipE8RrPGaOsCrGX1cCcZSn9ngOsKDO8o/view?usp=sharing
val_data_fn = 'checkpoints/cpsc_val_data.pth'


def load_params(model, ckpt, device):
    params_dict = torch.load(ckpt, device)
    model.load_state_dict(params_dict['model_state_dict'])
    return model.to(device).eval()


def load_dataset(datafolder: str = './data/CPSC/') -> (DataLoader, int):
    if os.path.exists(test_data_fn) is False or os.path.exists(val_data_fn) is False:
        _, val_dataloader, test_dataloader, num_classes = load_datasets(datafolder=datafolder,
                                                                        experiment='rhythm',
                                                                        batch_size=1)
        torch.save(test_dataloader, test_data_fn)
        torch.save(val_dataloader, val_data_fn)
    else:
        test_dataloader = torch.load(test_data_fn)
        val_dataloader = torch.load(val_data_fn)
    return val_dataloader, test_dataloader


def norm(ecg_signal):
    min_val = np.min(ecg_signal)
    max_val = np.max(ecg_signal)
    normalized_ecg = (ecg_signal - min_val) / (max_val - min_val)
    return normalized_ecg


def normalize_data(ecg_data):
    ecg_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
    ecg_data = 2 * ecg_data - 1  # Normalize to -1 to 1
    return ecg_data


def paint_RR_interval(ax, rPeak, fc, ec):
    for idx, x in enumerate(rPeak):
        if idx % 2 == 1:
            continue
        if idx == 0:
            highlight_start = 0
            highlight_end = x
        else:
            highlight_start = rPeak[idx - 1]
            highlight_end = x
        ax.axvspan(highlight_start, highlight_end, facecolor=fc[idx % 2], edgecolor=ec[idx % 2], linewidth=1, alpha=1,
                   clip_on=True)

    # overlay border
    for idx, x in enumerate(rPeak):
        if idx % 2 == 0:
            continue
        if idx == 0:
            highlight_start = -20
            highlight_end = x
        else:
            highlight_start = rPeak[idx - 1]
            highlight_end = x
        ax.axvspan(highlight_start, highlight_end, facecolor=fc[idx % 2], edgecolor=ec[idx % 2], linewidth=1, alpha=1,
                   clip_on=True)


def detectRPeakAndShowECG():
    val_dataloader, test_dataloader = load_dataset()
    name_type = ['I-AVB', 'AF', 'LBBB', 'RBBB', 'NORM', 'PAC', 'STD', 'STE', 'PVC']
    name_lead = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i, (ecg, label) in enumerate(test_dataloader):
        ecg = ecg[0].cpu().numpy()
        label = label[0].cpu().numpy()
        # PVC disease and single label
        if label[-1] != 1 or label.sum() != 1:
            continue

        sampling_rate = 100
        rPeak = ECGTool.christov_segmenter(ecg[0], sampling_rate=100)[0] / sampling_rate  # use Lead I ECG signal
        rPeak = np.append(rPeak, 10)
        fig, axes = plt.subplots(6, 2, figsize=(15, 7), sharex=True, sharey=True)
        idx_lead = 0
        for col in range(2):
            for row in range(6):
                lead_data = ecg[idx_lead]
                time = np.linspace(0, 10, len(lead_data))
                axes[row][col].plot(time, lead_data, c='black')
                axes[row][col].set_xlim(time[0] - 0.1, time[-1] + 0.1)
                axes[row][col].set_ylabel(name_lead[idx_lead], rotation=0, labelpad=15, fontsize=14)
                axes[row][col].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                colors = ['#dbe6ee', '#b8d1e2']  # face color
                edge_color = ['#aecade', '#78a9cc']  # edge color

                paint_RR_interval(axes[row][col], rPeak, colors, edge_color)

                for spine in axes[row][col].spines.values():
                    spine.set_visible(False)
                idx_lead += 1
        name_class = ', '.join([f'{name_type[idx]}' for idx in range(len(label)) if label[idx] == 1])
        print(name_class)
        plt.subplots_adjust(hspace=0, wspace=0.06, top=0.98, bottom=0.02, left=0.03, right=0.99)
        plt.show(dpi=300)


def detectRPeakAndShowMultiECG():
    val_dataloader, test_dataloader = load_dataset()
    name_type = ['I-AVB', 'AF', 'LBBB', 'RBBB', 'NORM', 'PAC', 'STD', 'STE', 'PVC']
    name_lead = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    target_idx = [57, 178, 74, 416, 589]  # RBBB NORM PVC
    label_idx = []
    collect_data = []
    for i, (ecg, label) in enumerate(test_dataloader):
        if i not in target_idx:
            continue
        ecg = ecg[0].cpu().numpy()
        label = label[0].cpu().numpy()
        collect_data.append(ecg)
        label_idx.append(label)

    sort_idx = [2, 3, 0, 1, 4]
    sort_data = [collect_data[i] for i in sort_idx]
    sort_label = [label_idx[i] for i in sort_idx]
    fig, axes = plt.subplots(len(collect_data), figsize=(14, 1.2*len(collect_data)), sharex=True, sharey=True)  # figsize=(15, 7), sharey=True
    for i, ecg in enumerate(sort_data):
        idx_lead = 7  # V1
        lead_data = ecg[idx_lead]
        time = np.linspace(0, 10, len(lead_data))

        axes[i].plot(time, lead_data, c='black', linewidth=2)
        axes[i].set_xlim(time[0] - 0.1, time[-1] + 0.1)
        name_class = '+'.join([f'{name_type[idx]}' for idx in range(len(sort_label[i])) if sort_label[i][idx] == 1])
        axes[i].set_ylabel(name_class, rotation=45, labelpad=10, fontsize=12)
        axes[i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        sampling_rate = 100
        rPeak = ECGTool.christov_segmenter(ecg[1], sampling_rate=100)[0] / sampling_rate  # use Lead II ECG signal
        rPeak = np.append(rPeak, 10)
        colors = ['#dbe6ee', '#b8d1e2']  # face color  <= default
        edge_color = ['#aecade', '#78a9cc']  # edge color

        paint_RR_interval(axes[i], rPeak, colors, edge_color)

        for spine in axes[i].spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(hspace=0.1, wspace=0, top=0.98, bottom=0.02, left=0.04, right=0.99)
    plt.show(dpi=300)


def showAttentionOfThreeStage():
    model = DB_ECGNet()
    load_params(model, ckpt, device)

    val_dataloader, test_dataloader = load_dataset()

    name_type = ['I-AVB', 'AF', 'LBBB', 'RBBB', 'NORM', 'PAC', 'STD', 'STE', 'PVC']
    name_lead = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    test_target_idx = [2, 237]
    val_target_idx = [400]
    for i, (ecg, label) in enumerate(val_dataloader):  # modify dataloader
        ecg = ecg.to(device)
        label = label[0].cpu().numpy()
        if i not in val_target_idx:
            continue

        with torch.no_grad():  # predict correct
            output = model(ecg)
            output = torch.sigmoid(output).cpu().detach().numpy()[0]
        threshold = 0.5
        pred = np.where(output >= threshold, 1, 0)
        if not np.array_equal(pred, label):
            continue

        attn_stage = []
        attn1 = model.block0[0].local2Global.attn[0].mean(0)[0].cpu().numpy()
        attn_stage.append(np.repeat(attn1, 2))

        attn2 = model.block1[1].local2Global.attn[0].mean(0)[0].cpu().numpy()
        attn_stage.append(np.repeat(attn2, 4))

        attn3 = model.block2[-1].local2Global.attn[0].mean(0)[0].cpu().numpy()
        attn_stage.append(np.repeat(attn3, 8))

        ecg = ecg[0].cpu().numpy()

        fig, axes = plt.subplots(4, 1, figsize=(14, 4), sharex=True, height_ratios=[1, 1, 1, 1])
        y_label = ['signal', 'layer 1', 'layer 4', 'layer 9']

        def normalize_data(ecg_data):
            ecg_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
            ecg_data = 2 * ecg_data - 1  # Normalize to -1 to 1
            return ecg_data

        for row in range(4):
            lead_data = ecg[1]  # Lead-V2
            time = np.linspace(0, 10, len(lead_data))
            if row == 0:
                axes[row].plot(time, normalize_data(lead_data), c='#6886b8', linewidth=2)
                axes[row].set_facecolor('#e6e6ed')
                axes[row].set_xticks([xi for xi in range(0, 11, 1)])
                axes[row].set_yticks(np.arange(-1.25, 1.25, 0.5))
                axes[row].grid(True, which='both', linestyle='-', linewidth=1, color='#FFFFFF')

            else:
                gradient = np.vstack((attn_stage[row - 1], attn_stage[row - 1]))
                axes[row].imshow(gradient, aspect='auto',
                                 extent=(time.min(), time.max(), lead_data.min(), lead_data.max()),
                                 cmap='Reds', zorder=-1)  # , filternorm=True  cmap="plasma"

            axes[row].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            axes[row].set_ylabel(y_label[row], rotation=90, labelpad=10, fontsize=12)
            for spine in axes[row].spines.values():
                spine.set_color('#333')

        name_class = '+'.join([f'{name_type[idx]}' for idx in range(len(label)) if label[idx] == 1])
        print(name_class)
        plt.subplots_adjust(hspace=0.1, wspace=0, top=0.98, bottom=0.02, left=0.03, right=0.99)
        plt.show(dpi=300)


def showAttentionOfLastLayer():
    model = DB_ECGNet()
    load_params(model, ckpt, device)

    val_dataloader, test_dataloader = load_dataset()

    name_type = ['I-AVB', 'AF', 'LBBB', 'RBBB', 'NORM', 'PAC', 'STD', 'STE', 'PVC']
    name_lead = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    test_target = [353, 389]  # PVC, PAC
    passIdx = 0
    for i, (ecg, label) in enumerate(test_dataloader):  # modify dataloader
        ecg = ecg.to(device)
        label = label[0].cpu().numpy()
        if i not in test_target:
            continue

        with torch.no_grad():  # predict correct
            output = model(ecg)
            output = torch.sigmoid(output).cpu().detach().numpy()[0]
        threshold = 0.5
        pred = np.where(output >= threshold, 1, 0)
        if not np.array_equal(pred, label):
            continue

        def normalize_data(ecg_data):
            ecg_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
            ecg_data = 2 * ecg_data - 1  # Normalize to -1 to 1
            return ecg_data

        attn_stage = []
        attn_head_num = 8
        for k in range(attn_head_num):
            attn = model.block2[-1].local2Global.attn[0][k][0].cpu().numpy()
            f1 = interp1d(np.linspace(0, 10, len(attn)), attn, kind='linear')
            attn = f1(np.linspace(0, 10, 1000))
            attn_stage.append(attn)

        ecg = ecg[0].cpu().numpy()
        y_label = [f'head-{i}' for i in range(1, attn_head_num + 1)]
        fig, axes = plt.subplots(attn_head_num, figsize=(14, attn_head_num), sharex=True)

        # adjust the display order of the pictures
        if passIdx == 0:
            sort_idx = [0, 1, 2, 4, 5, 6, 3, 7]
            passIdx += 1
        else:
            sort_idx = [0, 1, 6, 7, 4, 3, 5, 2]
        attn_heads = [attn_stage[idx] for idx in sort_idx]
        # Lead-V2
        lead_idx = 1
        for row in range(attn_head_num):
            lead_data = ecg[lead_idx]
            time = np.linspace(0, 10, len(lead_data))
            axes[row].plot(time, normalize_data(lead_data), c='#277bb5', linewidth=2)
            gradient = np.vstack((attn_heads[row], attn_heads[row]))
            axes[row].imshow(gradient, aspect='auto',
                             extent=(time.min(), time.max(), -1.5, 1.5),
                             cmap='Reds', zorder=-1)

            axes[row].text(0, 0, f"{name_lead[lead_idx]}", fontsize=12, color='black',
                           verticalalignment='bottom', horizontalalignment='left', transform=axes[row].transAxes)
            axes[row].set_ylim(-1.25, 1.25)
            axes[row].set_yticks(np.arange(-1.25, 1.25, 0.5))
            axes[row].set_xticks([i for i in range(1, 11)])
            axes[row].grid(True, which='major', linestyle='--', linewidth=1, color='#BABABA')
            axes[row].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            axes[row].set_ylabel(y_label[row], rotation=90, labelpad=7, fontsize=12)
            for spine in axes[row].spines.values():
                spine.set_color('#333')
        name_class = '+'.join([f'{name_type[idx]}' for idx in range(len(label)) if label[idx] == 1])
        print(name_class)
        plt.subplots_adjust(hspace=0.05, wspace=0, top=0.98, bottom=0.02, left=0.03, right=0.99)
        plt.grid(True)
        plt.show(dpi=300)


def showIGAttributeMultiClassFig():
    model = DB_ECGNet()
    load_params(model, ckpt, device)

    val_dataloader, test_dataloader = load_dataset()

    ig = IntegratedGradients(model)

    name_type = ['I-AVB', 'AF', 'LBBB', 'RBBB', 'NORM', 'PAC', 'STD', 'STE', 'PVC']
    name_lead = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    ig_arr = []
    ecg_arr = []
    label_arr = []
    target_idx = [3, 57, 43, 74, 32, 166]
    for i, (ecg, label) in enumerate(test_dataloader):
        if i not in target_idx:
            continue
        ecg = ecg.to(device)
        label = label[0].cpu().numpy()
        label_arr.append(label)
        # get IG-attribute
        attr_arr = []
        for tg, lb in enumerate(label):
            if lb != 1:
                continue
            attr = ig.attribute(ecg, target=tg, n_steps=100)
            attr_arr.append(attr.detach().cpu().numpy())
        attr_arr = np.array(attr_arr)
        attr_arr = attr_arr.squeeze(1)  # [k, 1, 12, 1000] => [k, 12, 1000]
        ecg = ecg[0].cpu()
        ig_arr.append(attr_arr)
        ecg_arr.append(ecg)

    # adjust the order of disease display
    sort_idx = [2, 5, 0, 3, 1, 4]
    sort_ig = [ig_arr[i] for i in sort_idx]
    sort_ecg = [ecg_arr[i] for i in sort_idx]
    sort_label = [label_arr[i] for i in sort_idx]

    target_lead = [[1, 7, 10], [1, 7, 10], [1, 7, 10], [1, 7, 10], [1, 7, 10], [1, 7, 10]]

    def min_max_normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def normalize_data(ecg_data):
        ecg_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
        ecg_data = 2 * ecg_data - 1  # Normalize to -1 to 1
        return ecg_data

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    fig, axes = plt.subplots(6, 3, figsize=(22, 7), width_ratios=[2, 1, 1])
    for i, lead_arr in enumerate(target_lead):
        for j, lead_idx in enumerate(lead_arr):
            if j > 0:
                lead_data = sort_ecg[i][lead_idx][:500].cpu().numpy()
                lead_ig = sort_ig[i][0][lead_idx][:500]
                time = np.linspace(0, 5, 500)
                axes[i][j].set_xticks([i for i in range(1, 6)])
            else:
                lead_data = sort_ecg[i][lead_idx].cpu().numpy()
                lead_ig = sort_ig[i][0][lead_idx]
                time = np.linspace(0, 10, 1000)
                axes[i][j].set_xticks([i for i in range(1, 11)])

            attn = np.abs(lead_ig)
            window_size = 5
            smoothed_attn = moving_average(min_max_normalize(attn), window_size)

            axes[i][j].plot(time, normalize_data(lead_data), c='#277bb5', linewidth=2)
            gradient = np.vstack((smoothed_attn, smoothed_attn))

            axes[i][j].set_ylim(-1.5, 1.5)
            axes[i][j].set_yticks(np.arange(-1.5, 1.5, 0.5))
            axes[i][j].imshow(gradient, aspect='auto',
                              extent=(time.min(), time.max(), -1.5, 1.5),
                              cmap='Reds', zorder=-1)  # , filternorm=True  cmap="plasma"

            axes[i][j].grid(True, which='major', linestyle='--', linewidth=1, color='#BABABA')
            axes[i][j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in axes[i][j].spines.values():
                spine.set_color('#333')

            axes[i][j].text(0, 0, f" {name_lead[lead_idx]}", fontsize=12, color='black',
                            verticalalignment='bottom', horizontalalignment='left', transform=axes[i][j].transAxes)
            if j == 0:
                axes[i][j].set_ylabel(name_type[sort_label[i].argmax()], rotation=45, labelpad=12, fontsize=12)

    plt.subplots_adjust(hspace=0.05, wspace=0.06, top=0.98, bottom=0.02, left=0.025, right=0.99)
    plt.legend()
    plt.show(dpi=300)


if __name__ == '__main__':
    """
        download test dataset
    """

    # detectRPeakAndShowECG()
    # showAttentionOfLastLayer()
    showAttentionOfThreeStage()
    # showIGAttributeMultiClassFig()

