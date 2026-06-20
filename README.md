# Comparison of Optimizers for Training Deep Neural Networks

**Case Study: ResNet-34 on CIFAR-100**

End-of-module project — Master of Excellence in Artificial Intelligence (M1)
Hassan II University of Casablanca — Faculty of Sciences Ben M'Sick

## Authors

- OUARRAK Layla
- OUTIGHLI Sanae
- RHAOUFAL Khadija

**Supervised by:** Prof. Faouzia BENABBOU

## Project Overview

This project experimentally compares **six optimizers** for training a deep neural network on an image classification task:

- SGD (Stochastic Gradient Descent)
- SGD with Momentum
- Adagrad
- RMSprop
- Adam
- Nadam

All optimizers are evaluated under strictly identical experimental conditions: same model architecture (ResNet-34), same dataset (CIFAR-100), same batch size, and same number of epochs. The goal is to compare their convergence speed, training stability, final accuracy, and computational cost (training time and GPU memory usage), and to provide practical recommendations depending on user constraints.


## Dataset

**CIFAR-100** (Krizhevsky, 2009): 60,000 color images of size 32×32 pixels across 100 classes (50,000 training images, 10,000 test images). The training set is further split into 45,000 training images and 5,000 validation images.

## Model

**ResNet-34** (He et al., 2016), adapted for CIFAR-100:
- Initial 7×7 convolution (stride 2) replaced with a 3×3 convolution (stride 1) to better suit the 32×32 input resolution
- Initial max-pooling layer removed
- Final fully-connected layer changed from 1,000 outputs (ImageNet) to 100 outputs (CIFAR-100)
- ~21.3 million trainable parameters

## Experimental Protocol

| Setting | Value |
|---|---|
| Batch size | 64 |
| Epochs | 10 |
| Loss function | Cross-Entropy |
| Gradient clipping | max norm = 1.0 |
| Hardware | NVIDIA Tesla T4 GPU |
| Framework | PyTorch 2.x, Torchvision |
| Random seed | 42 (fixed for reproducibility) |

### Hyperparameters per optimizer

| Optimizer | Hyperparameters |
|---|---|
| SGD | lr=0.01, momentum=0, weight_decay=5e-4 |
| SGD + Momentum | lr=0.01, momentum=0.9, weight_decay=5e-4 |
| Adagrad | lr=0.01, eps=1e-10 |
| RMSprop | lr=0.0005, alpha=0.9, eps=1e-8, weight_decay=1e-4 |
| Adam | lr=0.001, betas=(0.9, 0.999), eps=1e-8 |
| Nadam | lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4 |

## Results Summary (10 epochs)

| Optimizer | Test Acc. (%) | Test Loss | Time/Epoch (s) | Peak GPU Mem. (MB) |
|---|---|---|---|---|
| SGD | 12.79 | 3.74 | 73.5 | 784 |
| SGD + Momentum | 39.60 | 2.29 | 74.7 | 963 |
| Adagrad | 47.31 | 1.98 | 75.3 | 1,222 |
| RMSprop | 49.09 | 2.00 | 76.3 | 1,137 |
| **Adam** | **57.61** | **1.57** | 76.8 | 1,137 |
| Nadam | 51.45 | 1.75 | 77.5 | 1,393 |

**Adam** achieves the best test accuracy within this 10-epoch training budget, combining fast convergence with a moderate computational cost. See the full report for a detailed discussion of convergence speed, generalization, and computational trade-offs.

## How to Reproduce

1. Open `Optimisation.ipynb` in Google Colab or Jupyter
2. Run the notebook sequentially:
   - **Step 1–2**: Dataset loading, preprocessing, and augmentation
   - **Step 3**: ResNet-34 model definition and adaptation for CIFAR-100
   - **Step 4**: Training and evaluation utilities (`MetricsTracker` class)
   - **Step 5**: The six optimizer experiments (one cell per optimizer)
3. Run the final cell to reload the six CSV files and generate the comparative figure (`optimizer_comparison.png`)

## Report

The full project report (LaTeX/PDF), including the literature review, detailed methodology, results, and discussion, is available separately as part of the course submission.

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *Technical Report, University of Toronto*.
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
- Dozat, T. (2016). Incorporating Nesterov Momentum into Adam. *ICLR Workshop*.
