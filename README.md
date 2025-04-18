# FusionFCN‑SemSeg

**RGB‑Depth Fusion for Semantic Segmentation**

This repository implements a dual‑stream Fully Convolutional Network (FCN) that fuses RGB and depth modalities to perform pixel‑level semantic segmentation on road scenes. The model leverages pretrained ResNet50 backbones in TensorFlow/Keras, custom convolutional blocks, and transposed convolutions for upsampling.

---

## Features

- **Dual‑stream architecture**: Separate ResNet50 backbones for RGB and depth inputs.
- **Custom fusion head**: Two Conv2D layers (128→256 filters) per stream with dropout, followed by concatenation and transposed convolution for upsampling.
- **Single‑modality baselines**: RGB‑only and depth‑only FCNs for performance comparison.
- **Training & evaluation**: Scripts to train for 10 epochs, evaluate on the test set, and visualize sample predictions.
- **Extra experiments**: Fine‑tuning last layers of ResNet50 and batch‑size adjustments with analysis of overfitting.

---

## Repository Structure

```
FusionFCN‑SemSeg/
├── FusionFCN‑SemSeg.ipynb     # Main Jupyter notebook with full implementation and results
└── dataset/
    ├── train/
    │   ├── rgb/
    │   ├── depth/
    │   └── label/
    ├── validation/
    │   ├── rgb/
    │   ├── depth/
    │   └── label/
    └── test/
        ├── rgb/
        ├── depth/
        └── label/
```

---

## Data Availability

The `dataset/` folder contains all the `.npy` files for training, validation, and testing. Due to size constraints, the data is managed via Git Large File Storage (LFS). To fetch the data after cloning:

```bash
git lfs install          # install Git LFS if you haven't already
git clone https://github.com/SharinganWarrior/FusionFCN‑SemSeg.git
cd FusionFCN‑SemSeg
git lfs pull             # download the actual dataset files
```

Alternatively, you can download the dataset ZIP from the latest [GitHub Release](https://github.com/SharinganWarrior/FusionFCN‑SemSeg/releases/latest) and extract it into the `dataset/` directory.

---

## Setup & Installation

1. Clone the repo
   ```bash
   git clone https://github.com/SharinganWarrior/FusionFCN‑SemSeg.git
   cd FusionFCN‑SemSeg
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python3 ‑m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install ‑r requirements.txt
   ```

---

## Usage

1. Launch Jupyter Notebook
   ```bash
   jupyter notebook FusionFCN‑SemSeg.ipynb
   ```

2. Run all cells to:
   - Load and preprocess the dataset (resize to 256×256, one‑hot encode labels).
   - Define and compile the fusion and single‑stream FCNs.
   - Train models for 10 epochs on the train/validation split.
   - Evaluate on the test set and print loss/accuracy.
   - Visualize 5 random test examples with ground truth vs. predictions.

3. Convert to PDF (if needed):
   - File → Download as → HTML, then print to PDF.

---

## Results

| Modality         | Test Accuracy (%) |
|------------------|-------------------|
| RGB‑only         | ~52.9             |
| Depth‑only       | ~37.0             |
| RGB+Depth Fusion | ~24.5             |

> **Observation**: Fusion performance dropped due to noisy depth input interfering with RGB features. Fine‑tuning experiments boosted training accuracy but reduced validation performance, indicating overfitting.

---

## References

- [TensorFlow ResNet50 API](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
- [TensorFlow tf.image.resize](https://www.tensorflow.org/api_docs/python/tf/image/resize)
- [Keras Conv2D & Conv2DTranspose](https://keras.io/api/layers/convolution_layers/)
- [Keras Dropout Layer](https://keras.io/api/layers/regularization_layers/dropout/)
- [Keras SGD Optimizer](https://keras.io/api/optimizers/sgd/)

