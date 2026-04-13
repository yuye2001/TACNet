## TACNet

Pytorch Code of TACNet for Cross-modality Person Re-Identification (ReID)

### Highlight

---

We propose TACNet for cross-modality person ReID. Our method uses two-stage adaptive feature alignment, which realizes robust inter-modality and intra-modality feature alignment, and effectively alleviates the modal gap in RGB-Infrared person re-identification.

### Important Notice
This repository **only publishes the core idea and basic framework** of our proposed TACNet.
The **complete full implementation code** (including all training details, optimized tricks, and full pipeline) will be **publicly released after the paper is officially accepted**.
*The code has been tested in Python 3.8, Pytorch = 1.1.0. 

#### 1. Datasets

- (1) SYSU-MM01 Dataset : A mainstream visible-infrared ReID benchmark captured by 4 RGB and 2 IR cameras, covering indoor and outdoor scenes. It contains 491 identities with 287,628 RGB images and 15,792 IR images, which well validates TACNet’s cross-modality alignment ability.

- (2) RegDB Dataset : A typical visible-thermal ReID dataset including 412 identities, each with 10 RGB and 10 IR images. It is randomly split equally into training and testing sets to evaluate the generalization of TACNet.
#### 2. Training

Train a model by

```python
python main.py 

You may need manually define the data path first.



