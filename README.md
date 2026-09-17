# Comparative Study of Deep Learning Architectures on CIFAR-10 with Feature Scaling \& Normalization Techniques

## 📌 Summary

This repository contains an exhaustive comparative empirical study evaluating three seminal Deep Convolutional Neural Network architectures—**AlexNet**, **VGG16**, and **ResNet50**—on the **CIFAR-10** image classification benchmark. The study investigates two intertwined deep learning dimensions:

1. **Architectural Evolution \& Depth**: Comparing shallow classic CNNs (AlexNet), deep stacked homogeneous convolutions (VGG16), and deep residual networks with skip connections (ResNet50).
2. **Impact of Input Feature Scaling \& Normalization**: Quantifying the empirical convergence, stability, loss dynamics, and classification performance across four data scaling regimes:

   * **Baseline (Raw MinMax / ToTensor)**
   * **Max-Absolute Scaling (MaxAbs)**
   * **Standardization (Z-Score Normalization)**
   * **Robust Scaling (Median \& Interquartile Range)**

## 

## 🔬 Dataset: CIFAR-10

The **CIFAR-10** dataset is composed of 60,000 $32 \\times 32$ color images across 10 mutually exclusive categories:

* **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
* **Split**: 50,000 training images (5,000 per class) and 10,000 testing images (1,000 per class).
* **Challenge**: Low spatial resolution ($32 \\times 32$) makes standard architectures originally crafted for ImageNet ($224 \\times 224$) prone to immediate dimensional collapse unless properly adapted.



## 🏗️ Architectural Modifications for CIFAR-10

Standard ImageNet models utilize aggressive early downsampling, which quickly reduces $32 \\times 32$ inputs to degenerate spatial resolutions. To accommodate low-resolution images, the architectures were adapted as follows:

```
+---------------------------------------------------------------------------------------------------+
| MODEL         | ORIGINAL IMAGENET STEM \& DOWNSAMPLING  | CIFAR-10 ADAPTED ARCHITECTURE             |
+---------------------------------------------------------------------------------------------------+
| AlexNet       | Conv 11x11, s=4, p=2 -> Pool 3x3, s=2  | Conv 3x3, s=1, p=1 -> Pool 2x2, s=2      |
|               | FC: 9216 -> 4096 -> 4096 -> 1000       | FC: 4096 -> 1024 -> 1024 -> 10           |
+---------------+----------------------------------------+------------------------------------------+
| VGG16         | 13 Conv (3x3) + 5 MaxPool layers       | 13 Conv (3x3, p=1) + 5 MaxPool (2x2, s=2)|
|               | FC: 25088 -> 4096 -> 4096 -> 1000      | FC: 512 -> 512 -> 10                     |
+---------------+----------------------------------------+------------------------------------------+
| ResNet50      | Conv 7x7, s=2, p=3 -> MaxPool 3x3, s=2 | Conv 3x3, s=1, p=1 (No Initial MaxPool)  |
|               | Bottlenecks: \[3, 4, 6, 3], FC: 2048    | Bottlenecks: \[3, 4, 6, 3], FC: 2048 -> 10|
+---------------------------------------------------------------------------------------------------+
```

### 1\. Modified AlexNet (`AlexNetCIFAR`)

* Replaced large $11 \\times 11$ and $5 \\times 5$ convolution kernels with smaller $3 \\times 3$ convolutions with stride 1 and padding 1 to preserve small spatial feature maps.
* Reconfigured feature maps: Conv1 ($3 \\to 64$), Conv2 ($64 \\to 192$), Conv3 ($192 \\to 384$), Conv4 ($384 \\to 256$), Conv5 ($256 \\to 256$).
* Reduced dense classification layer dimensions from 4096 to 1024 units with Dropout ($p=0.5$) to prevent severe parameter explosion and overfitting.

### 2\. Modified VGG16 (`VGG16\_CIFAR`)

* Preserved the 5 convolutional blocks consisting of 13 convolutional layers with $3 \\times 3$ receptive fields and ReLU activations.
* Replaced the large $4096 \\times 4096$ fully connected classifier with a lightweight two-layer MLP ($512 \\to 512 \\to 10$) attached directly to the global/flattened feature output, drastically reducing parameter count while preserving deep spatial representation.

### 3\. Modified ResNet50 (`ResNet50CIFAR`)

* Modified the initial stem: Replaced the $7 \\times 7$ convolution (stride 2) and $3 \\times 3$ max-pooling with a single $3 \\times 3$ convolution (stride 1, padding 1) with Batch Normalization.
* Retained the 4 deep residual stages with **Bottleneck blocks** ($\[1\\times 1, 3\\times 3, 1\\times 1]$ convolutions with expansion factor of 4) configured with layer counts $\[3, 4, 6, 3]$ (50 layers total).
* Utilized Adaptive Average Pooling followed by a direct linear classification head ($2048 \\to 10$).



## ⚙️ Feature Scaling \& Normalization Methods

Input normalization ensures that input gradients remain well-conditioned, avoiding vanishing/exploding gradients and accelerating gradient descent across uneven loss surfaces:

|Scaling Technique|Mathematical Formulation|Characteristics \& Purpose|
|-|-|-|
|**Baseline (MinMax)**|$X\_{scaled} = \\frac{X - \\min(X)}{\\max(X) - \\min(X)} \\equiv \\frac{X}{255} \\in \[0, 1]$|Standard PyTorch `ToTensor()` conversion. Keeps pixel values in $\[0, 1]$ range without mean-centering.|
|**Max-Absolute Scaling (MaxAbs)**|$X\_{scaled} = \\frac{X}{\\max(\|X\|)}$|Scales features by the absolute maximum pixel value in the channel/batch, preserving strict sparsity.|
|**Standardization (Z-Score)**|$X\_{norm} = \\frac{X - \\mu}{\\sigma}$ <br> $\\mu = \[0.4914, 0.4822, 0.4465]$ <br> $\\sigma = \[0.2470, 0.2435, 0.2616]$|Shifts the distribution to zero mean ($\\mu = 0$) and unit variance ($\\sigma = 1$) using dataset-wide channel statistics.|
|**Robust Scaling**|$X\_{robust} = \\frac{X - Q\_2(X)}{Q\_3(X) - Q\_1(X)} = \\frac{X - \\text{Median}}{\\text{IQR}}$|Normalizes using median and Interquartile Range ($75^{\\text{th}} - 25^{\\text{th}}$ percentile), offering resistance to statistical outliers.|



## 🛠️ Experimental Setup \& Hyperparameters

All models were evaluated under uniform training configurations to ensure fair and unbiased comparative analysis:

* **Framework**: PyTorch \& Torchvision
* **Epochs**: 15
* **Batch Size**: 64
* **Optimizer**: Adam ($\\text{learning rate} = 10^{-3}$, $\\beta\_1 = 0.9$, $\\beta\_2 = 0.999$, $\\epsilon = 10^{-8}$)
* **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss()`)
* **Evaluation Metrics**:

  * Top-1 Accuracy (\%)
  * Precision (Macro \& Weighted average)
  * Recall (Macro \& Weighted average)
  * F1-Score (Macro \& Weighted average)
  * Cross-Entropy Loss
  * Multiclass Confusion Matrices (10 classes)

## 

## 📊 Comprehensive Results \& Metrics

Below is the consolidated performance summary across all 12 experimental combinations:

### 

### Overall Comparative Matrix

|Architecture|Normalization / Scaling|Test Accuracy (%)|Precision|Recall|F1-Score|Final Cross-Entropy Loss|
|-|-|:-:|:-:|:-:|:-:|:-:|
|**AlexNet**|Baseline|**71.31%**|0.7292|0.7131|0.7135|0.7365|
|**AlexNet**|Max-Absolute Scaling|**75.68%**|0.7586|0.7568|0.7562|0.5076|
|**AlexNet**|**Standardization**|**78.04%**|**0.7811**|**0.7804**|**0.7803**|**0.3538**|
|**AlexNet**|Robust Scaling|**73.83%**|0.7371|0.7383|0.7354|0.5924|
||||||||
|**VGG16**|Baseline|**74.96%**|0.7489|0.7496|0.7482|**0.1024**|
|**VGG16**|Max-Absolute Scaling|**76.00%**|0.7694|0.7600|0.7617|0.3917|
|**VGG16**|**Standardization**|**79.16%**|**0.7915**|**0.7916**|**0.7901**|0.1885|
|**VGG16**|Robust Scaling|**76.38%**|0.7711|0.7638|0.7635|0.2912|
||||||||
|**ResNet50**|Baseline|**83.09%**|0.8335|0.8309|0.8296|0.0887|
|**ResNet50**|**Max-Absolute Scaling**|**83.50%**|**0.8390**|**0.8350**|**0.8349**|**0.0797**|
|**ResNet50**|Standardization|**83.32%**|0.8366|0.8332|0.8330|0.0872|
|**ResNet50**|Robust Scaling|**80.66%**|0.8146|0.8066|0.8083|0.0871|

\---

## 

## 📈 Detailed Architectural Comparisons

### 1\. AlexNet Performance Breakdown

```
Baseline:         71.31%  \[Loss: 0.7365]
Robust Scaling:   73.83%  \[Loss: 0.5924]  (+2.52%)
MaxAbs Scaling:   75.68%  \[Loss: 0.5076]  (+4.37%)
Standardization:  78.04%  \[Loss: 0.3538]  (+6.73%) ★ Best for AlexNet
```

* **Analysis**: Because AlexNet lacks internal Batch Normalization layers in its original formulation, it relies heavily on well-conditioned input data. **Standardization produces a massive +6.73% accuracy improvement** and cuts the loss by more than half ($0.7365 \\to 0.3538$).

### 2\. VGG16 Performance Breakdown

```
Baseline:         74.96%  \[Loss: 0.1024]
MaxAbs Scaling:   76.00%  \[Loss: 0.3917]  (+1.04%)
Robust Scaling:   76.38%  \[Loss: 0.2912]  (+1.42%)
Standardization:  79.16%  \[Loss: 0.1885]  (+4.20%) ★ Best for VGG16
```

* **Analysis**: VGG16's 13 homogeneous convolutional layers extract rich hierarchical representations. Standardization once again leads the pack at **79.16% accuracy**, demonstrating how zero-centering prevents activation saturation across deeply cascaded non-linearities.

### 3\. ResNet50 Performance Breakdown

```
Robust Scaling:   80.66%  \[Loss: 0.0871]
Baseline:         83.09%  \[Loss: 0.0887]
Standardization:  83.32%  \[Loss: 0.0872]  (+0.23%)
MaxAbs Scaling:   83.50%  \[Loss: 0.0797]  (+0.41%) ★ Best for ResNet50
```

* **Analysis**: ResNet50 achieves the overall highest accuracy (**83.50%**) and lowest loss (**0.0797**). Thanks to internal **Batch Normalization** in every residual bottleneck, ResNet50 exhibits remarkable robustness across different input scaling methods, retaining $>83%$ accuracy across Baseline, MaxAbs, and Standardization.



## 🧠 Key Observations \& Theoretical Insights

### 1\. Depth \& Residual Learning (ResNet50 > VGG16 > AlexNet)

* **Degradation Problem Solved**: Without residual shortcuts, plain deep networks suffer from vanishing gradients and optimization stagnation. ResNet50 overcomes this through identity shortcut mappings $\\mathcal{H}(x) = \\mathcal{F}(x) + x$, enabling gradients to propagate directly through all 50 layers.
* **Top Accuracy**: ResNet50 achieved **83.50%** accuracy within just 15 epochs, outperforming VGG16 (79.16%) and AlexNet (78.04%).

### 2\. Why Standardization Dominates Shallow \& Deep Plain Networks

* **Zero-Centered Activations**: In architectures without extensive internal batch normalization (e.g., AlexNet, VGG), shifting input distribution to $\\mu=0, \\sigma=1$ centers the activation distributions around the active, non-saturated linear region of ReLU and keeps weight updates isotropic.
* **Accuracy Jump**: Delivered **+6.73%** gain in AlexNet and **+4.20%** in VGG16.

### 3\. Max-Absolute Scaling vs. Baseline

* MaxAbs maintains strict sparsity (zero values remain zero) while bounding all inputs to $\[-1, 1]$ or $\[0, 1]$. On ResNet50, MaxAbs attained the lowest test cross-entropy loss ($0.0797$) and top accuracy ($83.50%$).

### 4\. Robust Scaling Behavior on CIFAR-10

* While Robust Scaling is effective for tabular data or datasets corrupted by extreme statistical outliers, on clean photographic image datasets like CIFAR-10, scaling via the median and IQR compresses the dynamic range of non-outlier pixel distributions. Consequently, Robust Scaling yielded lower accuracy than Standardization across all models (e.g., 80.66% vs 83.32% on ResNet50).

### 5\. Confusion Matrix \& Class-Level Dynamics

* Across all models, **Automobiles**, **Ships**, and **Frogs** consistently exhibited high classification accuracy ($>85-90%$).
* Mutual confusion remained highest between semantically and visually similar categories:

  * **Cat $\\leftrightarrow$ Dog** (shared quadrupedal posture and texture features)
  * **Bird $\\leftrightarrow$ Airplane** (shared sky backgrounds and winged silhouettes)
  * **Automobile $\\leftrightarrow$ Truck** (shared wheeled geometry and metallic textures)



## 📊 Visual Results Gallery

|Model|Loss Curves|Accuracy Curves|Best Confusion Matrix|
|-|-|-|-|
|**AlexNet**|[Loss vs Epoch](AlexNet%20Results/Loss%20vs%20Epoch.png)|[Accuracy vs Epoch](AlexNet%20Results/Accuracy%20vs%20Epoch.png)|[Standardization CM](AlexNet%20Results/Standardization/Confusion%20Matrix.png)|
|**VGG16**|[Loss vs Epoch](VGG16%20Results/Loss%20vs%20Epoch.png)|[Accuracy vs Epoch](VGG16%20Results/Accuracy%20vs%20Epoch.png)|[Standardization CM](VGG16%20Results/Standardization/Screenshot%202026-03-26%20182118.png)|
|**ResNet50**|[Loss vs Epoch](ResNet50%20Results/Loss%20vs%20Epoch.png)|[Accuracy vs Epoch](ResNet50%20Results/Accuracy%20vs%20Epoch.png)|[MaxAbs CM](ResNet50%20Results/MaxAbs/Screenshot%202026-03-26%20181350.png)|



## 🚀 How to Run \& Reproduce

### 1\. Prerequisites

Ensure you have Python 3.9+ and the following packages installed:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn jupyter
```

### 2\. Running the Experiments

Launch Jupyter Notebook and open any of the experiment notebooks inside the `Code/` directory:

```bash
jupyter notebook Code/
```

* **AlexNet**: Run `Code/AlexNetCIFAR10.ipynb`
* **ResNet50**: Run `Code/ResNet50CIFAR10.ipynb`
* **VGG16**: Run `Code/VGG16CIFAR10.ipynb`

Each notebook self-contains:

1. Model class definition adapted for CIFAR-10.
2. Custom transform implementations (`ToTensorRaw`, `MaxAbsScaleTransform`, `Standardization`, `RobustScaleTransform`).
3. 15-epoch training loop using Adam optimizer.
4. Evaluation routines computing Accuracy, Precision, Recall, F1-Score, and Loss.
5. Automated generation of Loss vs Epoch, Accuracy vs Epoch, and Confusion Matrix plots.



## 🏁 Conclusion

|Goal|Finding|Best Recommendation|
|-|-|-|
|**Top Accuracy \& Robustness**|ResNet50 with Residual Bottlenecks outperforms shallower plain CNNs by up to **+12.19%** over baseline AlexNet.|**ResNet50**|
|**Optimal Input Preprocessing**|**Standardization (Z-Score)** consistently improves optimization dynamics and accuracy for models without heavy internal normalization.|**Standardization** ($\\mu, \\sigma$)|
|**Sparsity-Preserving Scaling**|**MaxAbs Scaling** delivers top performance when paired with modern BatchNorm networks.|**Max-Absolute Scaling**|



