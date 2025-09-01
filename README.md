### DualBranch-ECGNet

#### Feature

 - The model has excellent detection capabilities.
 - The model is interpretable.
 - The model is computationally efficient. ViT only requires a very small sequence of initialization tokens.


#### ECG signal

<img src="image\CPSC_AF_RBBB.png" alt="-" style="zoom: 40%;" />

⭐ Waveform and rhythm recognition are both very important.

#### Model Architecture

<img src="image\model.png" alt="-" style="zoom: 44%;" />

```python
CNN branch: => Local aggregation  # Waveform feature
ViT backbone: => Global aggregation  # Rhythm feature
Cross Attention: => Feature fusion
```

Unidirectional fusion can effectively improve the recall rate and prevent local waveform features from being confused by global fusion. Compared with bidirectional fusion, it can reduce the computational load and improve the effect of the attention heatmap.

#### *Multi-Scale Inverted Residual Block*

<img src="image\ms_irb.png" alt="-" style="zoom: 30%;" />

### Usage

- **Step 1**

```python
# data preprocessing
python data_convert.py  # CPSC
python data_partition.py  # Chapman
```

- **Step 2**

```python
python main.py --model DBECGNet --dataset CPSC
```

### Visualization

The visualization features are consistent with the disease features, and the model has good interpretability.

#### **Attention Heatmaps for the Three Stages of the Model**

*They are normal, PVC and PAC in sequence*

<img src="image\attn_three_stage_norm.png" alt="NORM" style="zoom:50%;" />

<img src="image\attn_three_stage_pvc.png" alt="PVC" style="zoom:50%;" />

<img src="image\attn_three_stage_pac.png" alt="PAC" style="zoom:50%;" />

#### **Attention heatmap of the last layer**

*They are normal, PVC and PAC in sequence*

<img src="image\attn_last_layer_norm.png" alt="ig_attr" style="zoom: 13%;" />

<img src="image\attn_last_layer_pvc.png" alt="attn_last_layer_pvc" style="zoom: 40%;" />

<img src="image\attn_last_layer_pac.png" alt="ig_attr" style="zoom: 40%;" />

Foreground and background isolation: Identification of key waveform features and global rhythms

#### **Integrated Gradients**

<img src="image\ig_attr.png" alt="ig_attr" />

#### Metrics

- CPSC dataset： f1: 0.84

### DataSet

- *CPSC:* [The China Physiological Signal Challenge 2018](http://2018.icbeb.org/Challenge.html)
- *PTB-XL:* [PTB-XL, a large publicly available electrocardiography dataset](https://www.physionet.org/content/ptb-xl/1.0.1/)
- *Chapman-Shaoxing:* [A 12-lead electrocardiogram database for arrhythmia research](https://figshare.com/collections/ChapmanECG/4560497/2)


### Acknowledge

Thanks for this open source framework [MVMS-net](https://github.com/ysxGitHub/MVMS-net)
