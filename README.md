# DeNAT Training Manual
## Deep Neurite Analysis Tool - Data Preparation and Model Training Guide

---

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Understanding the DeNAT Architecture](#understanding-the-denat-architecture)
4. [Data Preparation](#data-preparation)
5. [Training Workflows](#training-workflows)
6. [Model Architecture Configuration](#model-architecture-configuration)
7. [Running Training Sessions](#running-training-sessions)
8. [Monitoring and Evaluation](#monitoring-and-evaluation)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

DeNAT (Deep Neurite Analysis Tool) is a deep learning platform designed to automatically measure neurite outgrowth after injury. This manual provides comprehensive guidance for preparing training data and training custom models for neurite analysis tasks.

### What You'll Learn
- How to prepare and organize image data for training
- Understanding different training workflows (WT/C-KO, Replated, Pattern classification)
- Configuring model architecture parameters
- Running training sessions and monitoring performance
- Evaluating and validating trained models

---

## System Requirements

### Software Dependencies
Install the following Python packages before training:

```bash
pip install tensorflow>=2.0
pip install numpy
pip install opencv-python
pip install scikit-image
pip install pandas
pip install scikit-learn
pip install scipy
pip install tqdm
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster training)
- **RAM**: Minimum 16GB, 32GB recommended
- **Storage**: At least 10GB free space for datasets and models

---

## Understanding the DeNAT Architecture

### Model Components

DeNAT uses a convolutional neural network with the following components:

1. **Convolutional Backbone**: 5-stage architecture with exponential filter growth
2. **Feature Extraction**: Side modules with progressive down-sampling
3. **Classification Head**: Average pooling, flattening, and dense layers

### Key Architecture Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `filter_scale` | Controls exponential filter growth (filters = 2^(filter_scale + n)) | Varies by task |
| `dense_size` | Number of neurons in penultimate dense layer | 128 |
| `input_shape` | Input image dimensions | [1024, 1360, 1] |
| `out_drop` | Dropout rate before classification | 0.2 |
| `generic` | Enable/disable side module concatenation | True/False |

---

## Data Preparation

### Image Requirements

#### Format Specifications
- **File Format**: PNG images
- **Dimensions**: 1024 × 1360 pixels (fixed)
- **Color Space**: Grayscale (single channel)
- **Bit Depth**: 8-bit or 16-bit recommended
- **File Naming**: Must follow specific patterns for label extraction

### Dataset Organization

DeNAT supports three classification tasks, each requiring specific directory structures:

#### 1. Wild-type vs C-KO Classification

```
training_data/
├── wild_type/
│   ├── wt_female_001.png
│   ├── wt_male_001.png
│   └── ...
└── c_ko/
    ├── cko_female_001.png
    ├── cko_male_001.png
    └── ...
```

**Filename Pattern**: Images should contain labels in filenames for regex extraction
- Wild-type: Include "wt" or "wild_type"
- C-KO: Include "cko" or "c_ko"
- Sex classification: Include "female" or "male"

#### 2. Wild-type vs Replated Classification

```
replate_training/
├── wild_type/
│   ├── wt_001.png
│   └── ...
└── replated/
    ├── replate_001.png
    └── ...
```

**Filename Pattern**:
- Wild-type: Include "wt"
- Replated: Include "replate"

#### 3. Pattern Classification

```
PatternTrainImages/
├── pattern/
│   ├── pattern_001.png
│   └── ...
├── unpatterned/
│   ├── unpatterned_001.png
│   └── ...
└── blank/
    ├── blank_001.png
    └── ...
```

**Classes**:
- **Pattern**: Images showing patterned neurite growth
- **Unpatterned**: Images showing non-patterned growth
- **Blank**: Control/background images

### Validation Dataset

Always create a separate validation dataset (recommended 15-20% of total data):

```
PatternValidationImages/
├── pattern/
├── unpatterned/
└── blank/
```

### Data Preprocessing Pipeline

DeNAT automatically applies the following preprocessing steps:

1. **Image Loading**: PNG files decoded as single-channel grayscale
2. **Normalization**: Pixel values rescaled to [0, 1] range using min-max normalization
3. **Edge Detection**: Sobel edge detection applied, results rescaled to [0, 1]
4. **Batching**: Images grouped into specified batch sizes with prefetching

### Data Augmentation (Training Only)

During training, the following augmentations are automatically applied:

- **Random Cropping**: Maximum 30×30 pixels removed (configurable)
- **Random Horizontal Flipping**: 50% probability
- **Random Vertical Flipping**: 50% probability
- **Dataset Shuffling**: Buffer size matches total image count

---

## Training Workflows

### Workflow 1: Wild-type vs C-KO Training

**Use Case**: Binary classification of wild-type vs C-KO samples with sex differentiation

**Training Configuration**:
- Batch Size: 32 samples
- Epochs: 300
- Learning Rate: 1e-4
- Label Smoothing: 0.1
- Output Classes: 2

**To Run**:
```python
from model.training import train_wtko

# Specify your data directory and output path
train_wtko(
    data_dir='path/to/wild_type_cko_data',
    output_model_path='models/wtko_model.h5'
)
```

---

### Workflow 2: Wild-type vs Replated Training

**Use Case**: Binary classification of wild-type vs replated samples

**Training Configuration**:
- Batch Size: 16 samples
- Epochs: 50
- Learning Rate: 1e-4
- Label Smoothing: 0.1
- Output Classes: 2

**To Run**:
```python
from model.training import train_replate

# Specify your data directory and output path
train_replate(
    data_dir='path/to/replate_data',
    output_model_path='models/replate_model.h5'
)
```

---

### Workflow 3: Pattern Classification Training

**Use Case**: Three-class classification (pattern/unpatterned/blank)

**Training Configuration**:
- Batch Size: 4 samples
- Epochs: 300
- Learning Rate: 0.001
- Label Smoothing: 0.05
- Output Classes: 3
- Dense Units: 128

**To Run**:
```python
from model.pattern_training import train_pattern_classifier

# Specify training and validation directories
train_pattern_classifier(
    train_dir='PatternTrainImages',
    validation_dir='PatternValidationImages',
    output_model_path='models/pattern_classifier.h5'
)
```

**Additional Features**:
- Auxiliary classification for sex and genotype categories
- Learning rate reduction on validation loss plateau (patience: 5 epochs)
- Model checkpointing to preserve best validation performance

---

## Model Architecture Configuration

### Custom Training Function

For custom training configurations, use the `train_model()` function:

```python
from model.training import train_model
from model.architecture import neurite_classifier
from model.data import prepare_dataset

# Prepare datasets
train_data = prepare_dataset(
    base_dir='path/to/training_data',
    batch_size=32,
    is_replate=False,
    is_training=True
)

val_data = prepare_dataset(
    base_dir='path/to/validation_data',
    batch_size=32,
    is_replate=False,
    is_training=False
)

# Create model
model = neurite_classifier(
    filter_scale=3,
    dense_size=128,
    input_shape=[1024, 1360, 1],
    out_drop=0.2,
    generic=True,
    output_classes=2,
    out_layers=1
)

# Train model
train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    epochs=100,
    learning_rate=1e-4,
    output_classes=2,
    model_path='models/custom_model.h5',
    train_log='logs/train',
    val_log='logs/validation'
)
```

### Architecture Parameters Explained

#### filter_scale
Controls the number of filters in convolutional layers exponentially:
- Formula: `filters = 2^(filter_scale + layer_number)`
- Example: filter_scale=3, layer 1 → 2^(3+1) = 16 filters
- Higher values = more complex feature extraction
- Recommended range: 2-4

#### dense_size
Number of neurons in the penultimate fully-connected layer:
- Default: 128
- Higher values = more capacity to learn complex patterns
- Too high may cause overfitting on small datasets
- Recommended range: 64-256

#### out_drop
Dropout rate applied before the final classification layer:
- Default: 0.2 (20% dropout)
- Helps prevent overfitting
- Recommended range: 0.1-0.5

#### generic
Controls side module concatenation:
- `True`: Concatenates all side module outputs (richer features)
- `False`: Uses only main pathway (simpler model)
- Use `True` for complex classification tasks

---

## Running Training Sessions

### Step-by-Step Training Process

#### Step 1: Prepare Your Data
1. Collect and organize microscopy images
2. Ensure all images are 1024×1360 pixels
3. Convert to grayscale PNG format
4. Organize into appropriate directory structure
5. Split into training (80%) and validation (20%) sets

#### Step 2: Verify Data Organization
```python
import os

# Check training data
train_dir = 'path/to/training_data'
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if f.endswith('.png')])
        print(f"{class_name}: {num_images} images")
```

#### Step 3: Choose Training Workflow
Select the appropriate workflow based on your classification task:
- Wild-type vs C-KO: Use `train_wtko()`
- Replated analysis: Use `train_replate()`
- Pattern classification: Use pattern training workflow

#### Step 4: Start Training
```python
# Example for pattern classification
from model.pattern_training import train_pattern_classifier

train_pattern_classifier(
    train_dir='PatternTrainImages',
    validation_dir='PatternValidationImages',
    output_model_path='models/my_pattern_model.h5'
)
```

#### Step 5: Monitor Progress
Training progress will be displayed in the console showing:
- Current epoch number
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

---

## Monitoring and Evaluation

### TensorBoard Integration

DeNAT automatically logs training metrics to TensorBoard. To visualize:

```bash
tensorboard --logdir=logs/
```

Then open your browser to `http://localhost:6006`

**Metrics to Monitor**:
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without diverging from training loss
- **Accuracy**: Should increase over epochs
- **Learning Rate**: Check if learning rate scheduling is working

### Identifying Overfitting

**Signs of Overfitting**:
- Training loss continues to decrease while validation loss increases
- Large gap between training and validation accuracy
- Model performs poorly on new test images

**Solutions**:
1. Increase dropout rate (e.g., from 0.2 to 0.3)
2. Reduce model complexity (lower `dense_size` or `filter_scale`)
3. Add more training data
4. Increase data augmentation
5. Use early stopping based on validation loss

### Model Checkpointing

Pattern training automatically saves the best model based on validation performance:
- Models are saved when validation loss improves
- Only the best-performing model is retained
- Checkpoint files are saved to the specified output path

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Out of Memory (OOM) Errors
**Symptoms**: Training crashes with CUDA OOM or memory allocation errors

**Solutions**:
- Reduce batch size (e.g., from 32 to 16 or 8)
- Reduce model complexity (lower `filter_scale` or `dense_size`)
- Close other GPU-intensive applications
- Use mixed precision training (TensorFlow 2.x)

```python
# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

#### Issue 2: Low Training Accuracy
**Symptoms**: Accuracy remains low (<60%) after many epochs

**Solutions**:
- Check data quality and labels
- Verify image preprocessing is correct
- Increase model complexity (higher `filter_scale`)
- Reduce learning rate (e.g., from 1e-4 to 1e-5)
- Train for more epochs
- Check for class imbalance in dataset

#### Issue 3: Image Loading Errors
**Symptoms**: Errors during data loading, "file not found" messages

**Solutions**:
- Verify all images are PNG format
- Check directory paths are correct
- Ensure filenames follow expected patterns
- Verify file permissions

```python
# Test image loading
from PIL import Image
import os

data_dir = 'path/to/data'
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.png'):
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except Exception as e:
                print(f"Error loading {file}: {e}")
```

#### Issue 4: Validation Loss Not Improving
**Symptoms**: Validation loss plateaus or oscillates

**Solutions**:
- Enable learning rate reduction (already enabled in pattern training)
- Check if validation set is representative of training set
- Increase training data diversity
- Review data augmentation settings

---

## Best Practices

### Data Collection
1. **Consistency**: Use consistent imaging parameters (magnification, exposure, contrast)
2. **Quality**: Ensure high-quality, focused images without artifacts
3. **Diversity**: Include diverse examples representing natural variation
4. **Balance**: Maintain balanced class distributions (similar number of images per class)
5. **Validation**: Keep validation set completely separate from training

### Data Annotation
1. **Accuracy**: Ensure labels are correct and consistent
2. **Documentation**: Document labeling criteria and edge cases
3. **Multiple Reviewers**: Have annotations verified by multiple experts
4. **Quality Control**: Regularly audit labeled data for errors

### Training Strategy
1. **Start Small**: Begin with a small subset to verify pipeline works
2. **Baseline First**: Establish baseline performance before optimization
3. **Incremental Changes**: Change one parameter at a time
4. **Document Experiments**: Keep detailed logs of configurations and results
5. **Version Control**: Use version control for code and model checkpoints

### Model Evaluation
1. **Hold-out Test Set**: Reserve 10-15% of data for final testing (never used in training)
2. **Cross-validation**: Consider k-fold cross-validation for small datasets
3. **Biological Validation**: Validate results with domain experts
4. **Error Analysis**: Examine misclassified examples to identify patterns

### Computational Efficiency
1. **Data Caching**: Enable dataset caching to speed up training
```python
train_data = prepare_dataset(...).cache()
```
2. **Prefetching**: Use prefetching to overlap data loading and training
3. **Mixed Precision**: Enable for faster training on modern GPUs
4. **Batch Size Optimization**: Find the largest batch size that fits in memory

---

## Appendix A: File Naming Conventions

### Wild-type vs C-KO Dataset
```
wt_female_001.png
wt_male_002.png
cko_female_001.png
cko_male_002.png
```

### Replated Dataset
```
wt_sample_001.png
replate_sample_001.png
```

### Pattern Dataset
```
pattern_experiment_001.png
unpatterned_control_001.png
blank_background_001.png
```

---

## Appendix B: Example Training Script

Complete example script for custom training:

```python
#!/usr/bin/env python3
"""
DeNAT Custom Training Script
"""

import os
from model.training import train_model
from model.architecture import neurite_classifier
from model.data import prepare_dataset

# Configuration
TRAIN_DIR = 'data/training'
VAL_DIR = 'data/validation'
MODEL_OUTPUT = 'models/my_neurite_model.h5'
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

# Prepare datasets
print("Preparing training dataset...")
train_data = prepare_dataset(
    base_dir=TRAIN_DIR,
    batch_size=BATCH_SIZE,
    is_replate=False,
    is_training=True
)

print("Preparing validation dataset...")
val_data = prepare_dataset(
    base_dir=VAL_DIR,
    batch_size=BATCH_SIZE,
    is_replate=False,
    is_training=False
)

# Create model
print("Building model architecture...")
model = neurite_classifier(
    filter_scale=3,
    dense_size=128,
    input_shape=[1024, 1360, 1],
    out_drop=0.2,
    generic=True,
    output_classes=NUM_CLASSES,
    out_layers=1
)

# Display model summary
model.summary()

# Train model
print(f"Starting training for {EPOCHS} epochs...")
train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    output_classes=NUM_CLASSES,
    model_path=MODEL_OUTPUT,
    train_log='logs/train',
    val_log='logs/validation'
)

print(f"Training complete! Model saved to {MODEL_OUTPUT}")
```

---

## Appendix C: Quick Reference

### Training Workflows Summary

| Workflow | Batch Size | Epochs | Learning Rate | Classes | Use Case |
|----------|-----------|--------|---------------|---------|----------|
| WT vs C-KO | 32 | 300 | 1e-4 | 2 | Wild-type vs C-KO classification |
| Replated | 16 | 50 | 1e-4 | 2 | Wild-type vs Replated samples |
| Pattern | 4 | 300 | 0.001 | 3 | Pattern/Unpatterned/Blank |

### Key File Locations

- **Training Scripts**: `model/training.py`, `model/pattern_training.py`
- **Data Loader**: `model/data.py`
- **Architecture**: `model/architecture.py`
- **Example Data**: `Example data/`
- **Example Output**: `Example_output/`

### Useful Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run TensorBoard
tensorboard --logdir=logs/

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Verify image dimensions
python -c "from PIL import Image; img = Image.open('image.png'); print(img.size)"
```

---

## Support and Resources

- **GitHub Repository**: https://github.com/mano2991/Neurite-Analysis-Tool
- **Web Interface**: https://neuriteanalysis.netlify.app
- **Workflow Guide**: See `Example_output/DeNAT Analysis Workflow.pdf`
- **Demo Video**: `DeNAT_Demo.MP4` in repository root

---

## License

DeNAT is released under the MIT License for academic and research purposes.

---

**Document Version**: 1.0
**Last Updated**: March 2026
**Author**: DeNAT Development Team
