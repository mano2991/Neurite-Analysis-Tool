# DeNAT Training Manual
## Deep Neurite Analysis Tool - Data Preparation and Model Training Guide
### For Pyramidotomy and Spinal Cord Injury Analysis

---

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Understanding the DeNAT Architecture](#understanding-the-denat-architecture)
4. [Biological Background and Application](#biological-background-and-application)
5. [Data Collection and Preparation](#data-collection-and-preparation)
6. [Training Workflow](#training-workflow)
7. [Model Architecture Configuration](#model-architecture-configuration)
8. [Running Training Sessions](#running-training-sessions)
9. [Monitoring and Evaluation](#monitoring-and-evaluation)
10. [Using the Web Application](#using-the-web-application)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Introduction

DeNAT (Deep Neurite Analysis Tool) is a machine-learning framework designed to provide high-sensitivity automated measurement of neurite outgrowth following neural injury. This manual provides comprehensive guidance for preparing training data and training custom models specifically for pyramidotomy and spinal cord injury paradigms.

### What Makes DeNAT Unique

DeNAT addresses critical limitations in existing neurite analysis tools:

1. **Region of Interest (ROI) Selection**: Unlike conventional tools, DeNAT allows users to define specific anatomical regions for analysis, enabling targeted quantification of post-injury axonal sprouting while excluding irrelevant structures.

2. **High-Sensitivity Detection**: DeNAT achieves perfect sensitivity (1.0), ensuring no regenerating axons are missed - a critical requirement in injury research where false negatives (missing regenerating fibers) are more detrimental than false positives.

3. **Patch-Based Classification**: Rather than whole-image classification or pixel-wise segmentation, DeNAT uses overlapping patch-based analysis to preserve fine structural details without memory constraints.

4. **Web-Based Accessibility**: Freely available at https://neuriteanalysis.netlify.app with no installation required.

### Validation Performance

DeNAT has been rigorously validated on pyramidotomy datasets:
- **Sensitivity**: 1.0 (detects all neurites)
- **Precision**: 0.948
- **False Discovery Rate (FDR)**: 0.052
- **Pearson Correlation with Manual Counting**: R = 0.9991 (p < 2.2e-16)
- **Analysis Time Reduction**: From 45-60 minutes per image (manual) to 1-2 minutes (automated)

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

- **GPU**: NVIDIA GPU with CUDA support (strongly recommended for high-resolution image training)
- **RAM**: Minimum 16GB, 32GB strongly recommended for processing 4250×3350 pixel images
- **Storage**: At least 20GB free space for datasets and models
- **Operating System**: Linux, macOS, or Windows with Python 3.7+

---

## Understanding the DeNAT Architecture

### Computational Approach: Patch-Based Binary Classification

DeNAT uses a **patch-based binary classification** approach rather than semantic segmentation or whole-image classification:

1. Each high-resolution image (4250 × 3350 pixels) is partitioned into overlapping **256 × 256 pixel patches**
2. Patches are extracted with a **128-pixel stride** (50% overlap)
3. Each patch is independently classified as either "neurite-positive" or "neurite-negative"
4. Positive detections are mapped back to original coordinates for spatial aggregation

**Why This Approach?**
- Preserves high-resolution details without downsampling artifacts
- Avoids memory constraints of encoder-decoder architectures on very large biological images
- Captures neurites straddling patch boundaries through overlap

### Architecture Components

#### 1. Input Processing
- Original resolution maintained: 4250 × 3350 pixels
- Grayscale conversion with color inversion
- Tiled into overlapping 256×256 patches with 128-pixel stride
- Creates >15,000 patches from 235 whole images (effective training library expansion)

#### 2. Convolutional Backbone
- **Paired Convolutions**: Each stage includes strided and non-strided convolutions
  - Strided convolution: 2×2 stride, halves spatial dimensions
  - First layer: 5×5 kernel, 4 filters
  - Subsequent layers: 3×3 kernel, filters doubling (4→8→16→32→64)

- **Multi-Scale Side Branches**: Key architectural innovation
  - Side branches generated after each main-path convolution via max-pooling
  - Each side branch: 3×3 convolution with 16 filters
  - Outputs resized to common dimension and merged with main path
  - **Purpose**: Integrates fine neurite details with broader contextual information

#### 3. Feature Aggregation and Classification
- 128-filter convolution after feature aggregation
- Global average pooling → 128-dimensional feature vector
- Fully connected layer: 128 nodes
- Dropout: 20% to reduce overfitting
- Output layer: 2 nodes with sigmoid activation

**Why Two Output Nodes for Binary Classification?**
This configuration, combined with label smoothing (targets 0.1 and 0.9 instead of 0 and 1), prevents over-saturation on a single class during early training and improves calibration.

### Post-Classification Aggregation

1. Individual patches classified by CNN
2. Positive detections mapped to original image coordinates
3. Overlapping regions: probability scores averaged across all covering patches
4. Final threshold applied (default = 0.5) to produce binary detection map
5. Neurite counts aggregated within user-defined spatial bins (e.g., 0-200 μm, 200-400 μm from midline)

---

## Biological Background and Application

### Pyramidotomy Model

DeNAT was developed and validated specifically for the **pyramidotomy model**, one of the most common spinal cord injury paradigms for studying corticospinal tract plasticity and axon regeneration.

#### Experimental Context

**Animals**: Wild-type C57BL/6J mice (male and female, 8-12 weeks old)

**Surgical Procedure**:
1. Cortical viral injections with AAV tracers into sensorimotor cortex
2. Pyramidotomy 7 days post-injection (unilateral pyramidal tract transection)
3. 8-week recovery period
4. Tissue collection and processing

**Tissue Processing**:
- Cervical spinal cord (C1-C4) isolated
- Fixed in 4% PFA overnight at 4°C
- Embedded in gelatin
- 50 μm transverse sections cut via vibratome
- Three sections per animal sampled (C2-C6, 1.5 mm spacing)

### Image Acquisition

**Microscopy Parameters**:
- **For Pyramidotomy Samples**: Leica STED microscope, 63× objective
  - Tile scans: 520 × 520 pixels per tile
  - Z-steps: 1 μm
  - Scan speed: 400 Hz
  - Laser power: 40%
  - Detector gain: 1,250 V

- **For Medullary Pyramids** (normalization): Zeiss ApoTome.2 microscope

**Post-Acquisition Processing**:
- Maximum intensity projections generated for all stacks (Fiji/ImageJ)
- Final image size: 4250 × 3350 pixels
- Grayscale, single-channel format

### Fiber Index Normalization

**Important Concept**: Because AAV tracing efficiency varies between animals, raw axon counts must be normalized.

**Procedure**:
1. Count total labeled axons in intact medullary pyramids (anatomical source of corticospinal tract)
2. Count crossing axons in spinal cord sections
3. Calculate **Fiber Index** = (Crossing axons) / (Pyramid count)

This ensures differences reflect biological regeneration, not labeling variability.

### Spatial Quantification Strategy

DeNAT quantifies neurite outgrowth in **distance-based spatial bins** from the midline:

**Standard Bins** (matching pyramidotomy literature):
- 0-200 μm
- 200-400 μm
- 400-600 μm
- 600-800 μm
- >800 μm

**Counting Method**: 10 μm-wide counting lines perpendicular to midline

These parameters are customizable in the GUI but defaults match established gold-standard methodology for direct comparison with published literature.

---

## Data Collection and Preparation

### Image Requirements

#### Format Specifications

| Parameter | Specification |
|-----------|---------------|
| File Format | PNG (preferred) or JPG |
| Dimensions | 4250 × 3350 pixels (original resolution maintained) |
| Color Space | Grayscale (single channel) |
| Bit Depth | 8-bit or 16-bit |
| Background | Signal-free black background (AAV-labeled sections) |

**Critical**: Do NOT downsample images to standard classification sizes (e.g., 224×224). DeNAT is designed to work with full-resolution confocal images.

### Dataset Collection

#### Dataset Size Requirements

Based on the validated DeNAT implementation:
- **Total Dataset**: 235 high-resolution pyramidotomy images
- **Training Set**: 185 images (80%)
- **Test Set**: 50 images (20%)
- **Separate Validation Set**: 42 images for benchmarking
- **Inter-Rater Analysis Set**: 30 images

**Patch Expansion**: 235 whole images → >15,000 patches during training

#### Data Organization

```
DeNAT_Training_Data/
├── training_images/
│   ├── positive/
│   │   ├── pyramidotomy_001.png
│   │   ├── pyramidotomy_002.png
│   │   └── ...
│   └── negative/
│       ├── background_001.png
│       ├── background_002.png
│       └── ...
├── validation_images/
│   ├── positive/
│   └── negative/
└── test_images/
    ├── positive/
    └── negative/
```

**Classification Categories**:
- **Positive (Neurite-present)**: Patches/images containing neurite fibers crossing midline
- **Negative (Background)**: Patches/images with minimal or no neurite outgrowth

### Ground Truth Establishment

**Gold-Standard Protocol**:
1. Three independent, blinded expert observers manually trace all visible neurite fibers
2. Each observer annotates all validation images independently
3. Consensus count used as reference standard
4. Observers instructed to trace only confidently identified fibers (conservative approach)

**Important Note**: Conservative manual annotation may omit extremely faint or low-contrast neurites that are nonetheless biologically relevant. DeNAT often detects these, appearing as "false positives" when compared to ground truth, but secondary review confirms many are legitimate faint neurites.

### Data Preprocessing Pipeline

DeNAT automatically applies the following during training:

1. **Image Loading**: PNG files loaded at full resolution
2. **Grayscale Conversion**: If not already grayscale
3. **Color Inversion**: Inverts pixel values for optimal feature extraction
4. **Patch Extraction**: 256×256 patches with 128-pixel stride
5. **Normalization**: Applied during model training

### Data Augmentation (Training Only)

To increase dataset diversity and prevent overfitting:

| Augmentation | Parameters |
|--------------|------------|
| Random Horizontal Flip | 50% probability |
| Random Vertical Flip | 50% probability |
| Random Translation | Up to 60 pixels along either axis |
| Fill Method | Black pixels (matches signal-free background) |
| Dataset Shuffling | Buffer size = total image count |

**Important**: Black padding does not introduce artifacts because AAV-labeled confocal sections have signal-free black backgrounds.

---

## Training Workflow

### Cross-Validation Strategy

DeNAT uses **5-fold cross-validation** to ensure robust model evaluation:

1. Dataset split into 5 equal parts
2. Each fold:
   - 4 parts (80%) for training
   - 1 part (20%) for testing
3. Model never evaluated on data it has seen during training
4. Results averaged across all folds

### Training Configuration

#### Core Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 300 | Sufficient for convergence on high-res data |
| **Batch Size** | 32 | Balance between memory and gradient stability |
| **Optimizer** | Adam | Adaptive learning rates for efficient training |
| **Learning Rate** | 0.0001 | Prevents overshooting in fine-tuning |
| **Loss Function** | Binary Cross-Entropy | Treats two outputs as independent binary predictions |
| **Label Smoothing** | 0.1/0.9 | Prevents overconfident predictions, improves calibration |
| **Dropout** | 0.2 (20%) | Before final classification layer |

#### Label Smoothing Details

**Standard Binary Labels**: 0 (negative) and 1 (positive)
**Label Smoothed Targets**: 0.1 (negative) and 0.9 (positive)

**Why Label Smoothing?**
- Prevents over-confident predictions on high-contrast confocal data
- Improves model calibration
- Reduces overfitting
- Balances convergence between two output nodes

**Empirical Evidence**:
- Without label smoothing: Training accuracy 99%, validation accuracy 82% (overfitting)
- With label smoothing: Training-validation gap narrowed, validation accuracy 87.5%

### Training Procedure Step-by-Step

#### Step 1: Prepare Your Dataset

1. **Collect Pyramidotomy Images**:
   - Acquire confocal z-stacks of AAV-traced spinal cord sections
   - Generate maximum intensity projections
   - Save as PNG files (4250×3350 pixels)

2. **Organize Directory Structure**:
   ```bash
   mkdir -p DeNAT_Training_Data/training_images/{positive,negative}
   mkdir -p DeNAT_Training_Data/validation_images/{positive,negative}
   ```

3. **Label and Sort Images**:
   - Have 3 independent experts manually count neurites
   - Establish consensus ground truth
   - Sort into positive (high sprouting) and negative (low/no sprouting) categories

#### Step 2: Verify Data Quality

```python
import os
from PIL import Image
import numpy as np

def verify_dataset(data_dir):
    """Verify image dimensions and format"""
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    # Check dimensions
                    if img.size != (3350, 4250):  # Note: PIL uses (width, height)
                        print(f"Warning: {file} has dimensions {img.size}, expected (3350, 4250)")
                    # Check grayscale
                    if img.mode != 'L':
                        print(f"Warning: {file} is not grayscale (mode: {img.mode})")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

verify_dataset('DeNAT_Training_Data/training_images')
```

#### Step 3: Configure Training Script

```python
#!/usr/bin/env python3
"""
DeNAT Training Script for Pyramidotomy Data
Based on validated architecture from manuscript
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# Configuration
TRAIN_DIR = 'DeNAT_Training_Data/training_images'
VAL_DIR = 'DeNAT_Training_Data/validation_images'
MODEL_OUTPUT = 'models/denat_pyramidotomy.h5'

# Training parameters (from manuscript)
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.0001
PATCH_SIZE = 256
STRIDE = 128
LABEL_SMOOTHING = 0.1  # Targets become 0.1 and 0.9

# Data augmentation parameters
MAX_SHIFT = 60  # pixels
```

#### Step 4: Implement Data Pipeline

```python
def create_patch_dataset(image_dir, batch_size=32, is_training=True):
    """
    Creates dataset of 256x256 patches from high-res images
    """
    def load_and_preprocess(image_path, label):
        # Load image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(img, tf.float32)

        # Normalize to [0, 1]
        img = img / 255.0

        # Invert colors (as per manuscript preprocessing)
        img = 1.0 - img

        return img, label

    def extract_patches(image, label):
        """Extract overlapping 256x256 patches with 128-pixel stride"""
        patches = tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, STRIDE, STRIDE, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [-1, PATCH_SIZE, PATCH_SIZE, 1])
        labels = tf.repeat(label, tf.shape(patches)[0])
        return patches, labels

    def augment(image, label):
        """Apply data augmentation"""
        if is_training:
            # Random flips
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

            # Random translation (up to 60 pixels)
            shift_x = tf.random.uniform([], -MAX_SHIFT, MAX_SHIFT, dtype=tf.int32)
            shift_y = tf.random.uniform([], -MAX_SHIFT, MAX_SHIFT, dtype=tf.int32)
            image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])

        return image, label

    # Load image paths
    positive_images = tf.io.gfile.glob(f'{image_dir}/positive/*.png')
    negative_images = tf.io.gfile.glob(f'{image_dir}/negative/*.png')

    all_images = positive_images + negative_images
    all_labels = [1] * len(positive_images) + [0] * len(negative_images)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))

    if is_training:
        dataset = dataset.shuffle(len(all_images))

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.flat_map(lambda img, lbl:
        tf.data.Dataset.from_tensor_slices(extract_patches(img, lbl))
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
```

#### Step 5: Build Model Architecture

```python
def build_denat_model(input_shape=(256, 256, 1)):
    """
    Builds custom CNN architecture as described in manuscript
    Multi-scale side-branch architecture
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution (5x5 kernel, 4 filters)
    x = layers.Conv2D(4, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Side branches list
    side_branches = []

    # Paired convolutions with doubling filters (4→8→16→32→64)
    filters = [4, 8, 16, 32, 64]
    pool_sizes = [16, 8, 4, 2]  # For side branches

    for i, num_filters in enumerate(filters):
        # Strided convolution (reduces spatial dimensions by half)
        x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same',
                         activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Non-strided convolution
        x = layers.Conv2D(num_filters, kernel_size=3, padding='same',
                         activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Create side branch (except last layer)
        if i < len(pool_sizes):
            side = layers.MaxPooling2D(pool_size=pool_sizes[i])(x)
            side = layers.Conv2D(16, kernel_size=3, padding='same',
                               activation='relu')(side)
            side_branches.append(side)

    # Resize and merge side branches
    target_size = x.shape[1:3]
    merged_sides = []
    for side in side_branches:
        resized = tf.image.resize(side, target_size)
        merged_sides.append(resized)

    # Concatenate main path with all side branches
    if merged_sides:
        x = layers.Concatenate()([x] + merged_sides)

    # Final convolution (128 filters)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer (128 nodes)
    x = layers.Dense(128, activation='relu')(x)

    # Dropout (20%)
    x = layers.Dropout(0.2)(x)

    # Output layer (2 nodes with sigmoid)
    outputs = layers.Dense(2, activation='sigmoid',
                          bias_initializer='zeros')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='DeNAT_CNN')

    return model
```

#### Step 6: Train the Model

```python
def train_denat():
    """Main training function"""

    # Create datasets
    print("Preparing training dataset...")
    train_dataset = create_patch_dataset(TRAIN_DIR, BATCH_SIZE, is_training=True)

    print("Preparing validation dataset...")
    val_dataset = create_patch_dataset(VAL_DIR, BATCH_SIZE, is_training=False)

    # Build model
    print("Building model architecture...")
    model = build_denat_model()
    model.summary()

    # Apply label smoothing to loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy', 'Precision', 'Recall']
    )

    # Callbacks
    callbacks = [
        TensorBoard(log_dir='logs/train', histogram_freq=1),
        ModelCheckpoint(
            MODEL_OUTPUT,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        )
    ]

    # Train model
    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Training complete! Best model saved to {MODEL_OUTPUT}")

    return model, history

if __name__ == '__main__':
    model, history = train_denat()
```

---

## Model Architecture Configuration

### Custom CNN vs ResNet50

The manuscript compared custom CNN against ResNet50 as a baseline:

| Metric | Custom CNN | ResNet50 | Rationale for Custom CNN |
|--------|-----------|----------|--------------------------|
| **Sensitivity** | 0.90 | 0.84 | Better detection of faint neurites |
| **Precision** | 0.89 | 0.86 | Fewer false positives |
| **F-Score** | 0.90 | 0.85 | Better overall balance |
| **FDR** | 0.11 | 0.13 | Lower false discovery rate |
| **AUC** | 0.846 | 0.82 | Better discriminative performance |

**Why Custom CNN Was Selected**:

1. **No Downsampling**: Processes high-resolution patches without aggressive compression
2. **Multi-Scale Integration**: Side branches capture fine details AND contextual information
3. **Computational Efficiency**: Simpler architecture than deep ResNet
4. **Task-Specific Design**: Optimized for thin, faint neurite extensions

### Architecture Parameters

| Parameter | Default Value | Function |
|-----------|---------------|----------|
| `input_shape` | (256, 256, 1) | Patch size for classification |
| `initial_filters` | 4 | Starting number of convolutional filters |
| `filter_growth` | Exponential (4→8→16→32→64) | Feature complexity increases with depth |
| `side_branch_filters` | 16 | Filters in each side branch |
| `dense_units` | 128 | Fully connected layer before output |
| `dropout_rate` | 0.2 | Regularization strength |
| `output_nodes` | 2 | Independent probabilities for positive/negative |

---

## Monitoring and Evaluation

### TensorBoard Integration

Monitor training in real-time:

```bash
tensorboard --logdir=logs/train
```

Navigate to `http://localhost:6006`

**Key Metrics to Monitor**:

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease without diverging from training
3. **Accuracy**: Should increase and plateau
4. **Precision**: Proportion of correct positive predictions
5. **Recall (Sensitivity)**: Proportion of actual positives detected

### Identifying Overfitting

**Warning Signs**:
- Training loss continues decreasing while validation loss increases
- Large gap between training and validation accuracy
- Model performs poorly on new test images

**Solutions**:
1. Increase dropout rate (e.g., 0.2 → 0.3)
2. Increase label smoothing (e.g., 0.1 → 0.15)
3. Add more training data
4. Reduce model complexity (fewer filters)

### Performance Metrics

#### Confusion Matrix Metrics

| Metric | Formula | DeNAT Target |
|--------|---------|--------------|
| **Sensitivity (Recall)** | TP / (TP + FN) | 1.0 (detect all neurites) |
| **Precision** | TP / (TP + FP) | >0.94 |
| **F-Score** | 2TP / (2TP + FP + FN) | >0.97 |
| **False Discovery Rate** | FP / (TP + FP) | <0.06 |

**Important**: In regeneration research, **false negatives are more critical than false positives**. Missing regenerating axons could lead to incorrect conclusions about treatment efficacy.

#### Benchmarking Against Manual Counting

**Correlation Analysis**:
- Pearson correlation coefficient (R)
- R² coefficient of determination
- Target: R > 0.99 with manual ground truth

**Bland-Altman Analysis**:
- Mean bias between automated and manual counts
- Limits of agreement (±1.96 SD)
- Target: Lower bias than inter-rater variability

---

## Using the Web Application

### Accessing DeNAT

**Web Interface**: https://neuriteanalysis.netlify.app

**Features**:
- No installation required
- Browser-based analysis
- All processing occurs locally (no data uploaded to servers)
- Data privacy: Images not stored or transmitted

### Workflow for Analysis

#### Step 1: Upload Image

- Drag and drop PNG or JPG image
- Supported formats: Maximum intensity projections from confocal microscopy
- Recommended: 4250×3350 pixels (optimized for this resolution)

#### Step 2: Define Midline

- Draw reference midline on image
- This establishes the coordinate system for distance measurements

#### Step 3: Select Region of Interest (ROI)

**Key Innovation**: Human-in-the-Loop approach
- Use anatomical expertise to exclude irrelevant regions
- Draw polygon around area of interest (e.g., gray matter, specific laminae)
- Only neurites within ROI will be quantified

#### Step 4: Run Automated Analysis

- Click "Analyze" to run CNN classification
- Processing time: 1-2 minutes per image
- Progress indicator shows analysis status

#### Step 5: Review Results

**Spatial Binning Output**:
Neurite counts displayed by distance from midline:
- 0-200 μm
- 200-400 μm
- 400-600 μm
- 600-800 μm
- >800 μm

**Visualization**:
- Detected neurites overlaid on original image
- Color-coded by distance bin
- Dashed vertical lines indicate bin boundaries

#### Step 6: Export Data

**Export Options**:
1. **CSV File**: Quantitative data for statistical analysis
   - Neurite counts per bin
   - Spatial coordinates
   - Total counts

2. **Labeled Image**: Visualization for publication/presentation
   - Neurites color-coded by distance
   - ROI highlighted
   - Scale bars

3. **Full Report**: Combined PDF with images and data

### Additional Tools

**Manual Counting Mode**:
- For validation or comparison
- Click-based neurite marking
- Counts recorded and exportable

**Image Adjustments**:
- Brightness/contrast controls
- Rotation for consistent orientation
- Zoom for detailed inspection

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Out of Memory (OOM) Errors During Training

**Symptoms**:
- Training crashes with CUDA OOM
- "ResourceExhaustedError" messages

**Solutions**:
```python
# Reduce batch size
BATCH_SIZE = 16  # or even 8

# Enable mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Reduce patch overlap (increases stride)
STRIDE = 256  # No overlap (instead of 128)
```

#### Issue 2: Model Not Converging (Loss Not Decreasing)

**Symptoms**:
- Loss remains high after many epochs
- Accuracy stuck around 50% (random guessing)

**Solutions**:
1. **Check Data Pipeline**:
```python
# Verify labels are correct
for images, labels in train_dataset.take(1):
    print("Batch shape:", images.shape)
    print("Labels:", labels.numpy())
    print("Label range:", tf.reduce_min(labels), tf.reduce_max(labels))
```

2. **Adjust Learning Rate**:
```python
# Try higher initial learning rate
LEARNING_RATE = 0.001  # Instead of 0.0001
```

3. **Verify Preprocessing**:
```python
# Check image inversion is applied
import matplotlib.pyplot as plt
for img, lbl in train_dataset.take(1):
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Label: {lbl[0]}")
    plt.show()
```

#### Issue 3: High False Positive Rate

**Symptoms**:
- Model detects many structures as neurites that aren't
- FDR > 0.1 on validation set

**Solutions**:
1. **Increase Specificity Through ROI Selection**:
   - Users can exclude artifact-prone regions via GUI
   - Tighten ROI boundaries

2. **Adjust Classification Threshold**:
```python
# Instead of default 0.5 threshold, use 0.6 or 0.7
predictions = model.predict(patches)
positive_patches = predictions[:, 1] > 0.6  # Higher threshold
```

3. **Add Post-Processing Filters**:
```python
# Filter out small isolated detections
from scipy import ndimage
labeled, num_features = ndimage.label(detection_map)
sizes = np.bincount(labeled.ravel())
mask = sizes > 50  # Remove small objects
detection_map = mask[labeled]
```

#### Issue 4: Poor Performance on Dense Neurite Regions

**Symptoms**:
- Undercounting in areas with many overlapping fibers
- Lower correlation with manual counts in high-density samples

**Solutions**:
1. **Reduce Patch Stride** (more overlap):
```python
STRIDE = 64  # Instead of 128 (75% overlap)
```

2. **Ensemble Predictions**:
```python
# Average predictions from multiple augmentations
predictions = []
for _ in range(5):
    aug_patches = augment(patches)
    pred = model.predict(aug_patches)
    predictions.append(pred)
final_pred = np.mean(predictions, axis=0)
```

#### Issue 5: Inconsistent Results Across Runs

**Symptoms**:
- Different neurite counts for same image across multiple analyses
- Variability in validation metrics

**Solutions**:
1. **Set Random Seeds**:
```python
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

2. **Disable Data Augmentation for Inference**:
```python
# Only use augmentation during training
test_dataset = create_patch_dataset(TEST_DIR, is_training=False)
```

---

## Best Practices

### Data Collection

1. **Imaging Consistency**:
   - Use identical microscope settings across all samples
   - Same magnification, laser power, detector gain
   - Consistent z-step size and tile overlap

2. **Biological Replicates**:
   - Include multiple animals per experimental group
   - Sample multiple sections per animal (e.g., 3 sections from C2-C6)
   - Ensure representation of biological variability

3. **Control Groups**:
   - Include sham-operated controls
   - Blank images (no sprouting) for negative class
   - High-sprouting positive controls

4. **Quality Control**:
   - Exclude images with:
     - Excessive background fluorescence
     - Tissue damage/tears
     - Out-of-focus regions
     - Saturated pixels

### Ground Truth Annotation

1. **Multi-Rater Consensus**:
   - Minimum 3 independent expert raters
   - Blinded to experimental conditions
   - Consensus protocol for discrepancies

2. **Annotation Guidelines**:
   - Define inclusion criteria (e.g., minimum fiber length, intensity threshold)
   - Document edge cases
   - Regular inter-rater reliability checks (Cohen's Kappa)

3. **Conservative vs Liberal Annotation**:
   - **Conservative**: Only trace high-confidence neurites (may miss faint fibers)
   - **Liberal**: Include all possible neurites (may include artifacts)
   - DeNAT is designed for liberal detection (high sensitivity)

### Training Strategy

1. **Start with Subset**:
   - Verify pipeline on 20-30 images first
   - Check for data loading errors, format issues
   - Confirm augmentation is working correctly

2. **Hyperparameter Tuning**:
   - Change one parameter at a time
   - Document all experiments in log files
   - Use validation set to select best configuration

3. **Cross-Validation**:
   - Always use k-fold cross-validation (k=5 recommended)
   - Report mean ± SD across folds
   - Ensures robustness to data splits

4. **Version Control**:
   - Track model checkpoints with descriptive names
   - Save training configuration with each model
   - Use git for code versioning

### Model Evaluation

1. **Multiple Metrics**:
   - Don't rely on accuracy alone
   - Report sensitivity, precision, F-score, FDR
   - Include correlation analysis with manual counts

2. **Biological Validation**:
   - Verify results with domain experts
   - Check that detected patterns match known biology
   - Confirm expected differences between experimental groups

3. **Error Analysis**:
   - Examine false positives and false negatives
   - Identify systematic errors (e.g., specific image regions)
   - Iterate on model or preprocessing based on error patterns

4. **Independent Test Set**:
   - Reserve 10-15% of data never used in training/validation
   - Only evaluate once on this set (final model performance)
   - Represents real-world deployment scenario

### Computational Efficiency

1. **GPU Utilization**:
```python
# Check GPU availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Enable memory growth (prevent OOM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

2. **Data Pipeline Optimization**:
```python
# Use AUTOTUNE for optimal performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.cache()  # Cache preprocessed data
```

3. **Batch Size Tuning**:
   - Find largest batch size that fits in GPU memory
   - Powers of 2 often work best (16, 32, 64)
   - Monitor GPU memory usage with `nvidia-smi`

---

## Appendix A: Complete Training Script Example

```python
#!/usr/bin/env python3
"""
DeNAT Complete Training Script for Pyramidotomy Analysis
Validated architecture matching manuscript specifications
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# ========== Configuration ==========
TRAIN_DIR = 'DeNAT_Training_Data/training_images'
VAL_DIR = 'DeNAT_Training_Data/validation_images'
MODEL_OUTPUT = 'models/denat_pyramidotomy_final.h5'
LOG_DIR = 'logs'

# Training parameters
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.0001
LABEL_SMOOTHING = 0.1

# Patch extraction
PATCH_SIZE = 256
STRIDE = 128

# Augmentation
MAX_SHIFT = 60

# Random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ========== Data Pipeline ==========

def create_dataset(image_dir, batch_size=32, is_training=True):
    """Create patch-based dataset from whole images"""

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(img, tf.float32) / 255.0
        img = 1.0 - img  # Color inversion
        return img, label

    def extract_patches(img, label):
        patches = tf.image.extract_patches(
            images=tf.expand_dims(img, 0),
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, STRIDE, STRIDE, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [-1, PATCH_SIZE, PATCH_SIZE, 1])
        labels = tf.repeat(label, tf.shape(patches)[0])
        return tf.data.Dataset.from_tensor_slices((patches, labels))

    def augment(img, label):
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            shift_x = tf.random.uniform([], -MAX_SHIFT, MAX_SHIFT, dtype=tf.int32)
            shift_y = tf.random.uniform([], -MAX_SHIFT, MAX_SHIFT, dtype=tf.int32)
            img = tf.roll(img, shift=[shift_y, shift_x], axis=[0, 1])
        return img, label

    # Load paths
    pos_imgs = tf.io.gfile.glob(f'{image_dir}/positive/*.png')
    neg_imgs = tf.io.gfile.glob(f'{image_dir}/negative/*.png')

    all_paths = pos_imgs + neg_imgs
    all_labels = [1.0] * len(pos_imgs) + [0.0] * len(neg_imgs)

    dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))

    if is_training:
        dataset = dataset.shuffle(len(all_paths), seed=SEED)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.flat_map(extract_patches)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ========== Model Architecture ==========

def build_denat_cnn(input_shape=(256, 256, 1)):
    """Custom CNN with multi-scale side branches"""

    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(4, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    side_branches = []
    filters_list = [4, 8, 16, 32, 64]
    pool_sizes = [16, 8, 4, 2]

    # Paired convolutions with side branches
    for i, num_filters in enumerate(filters_list):
        # Strided convolution
        x = layers.Conv2D(num_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Non-strided convolution
        x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Side branch
        if i < len(pool_sizes):
            side = layers.MaxPooling2D(pool_sizes[i])(x)
            side = layers.Conv2D(16, 3, padding='same', activation='relu')(side)
            side_branches.append(side)

    # Merge side branches
    target_h, target_w = x.shape[1], x.shape[2]
    resized_sides = []
    for side in side_branches:
        resized = layers.Resizing(target_h, target_w)(side)
        resized_sides.append(resized)

    x = layers.Concatenate()([x] + resized_sides)

    # Final convolution
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='sigmoid', bias_initializer='zeros')(x)

    return models.Model(inputs, outputs, name='DeNAT_CNN')

# ========== Training ==========

def train():
    """Main training function"""

    print("=== DeNAT Training Pipeline ===\n")

    # Create datasets
    print(f"Loading training data from {TRAIN_DIR}...")
    train_ds = create_dataset(TRAIN_DIR, BATCH_SIZE, is_training=True)

    print(f"Loading validation data from {VAL_DIR}...")
    val_ds = create_dataset(VAL_DIR, BATCH_SIZE, is_training=False)

    # Build model
    print("\nBuilding model architecture...")
    model = build_denat_cnn()
    model.summary()

    # Compile
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

    # Callbacks
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True),
        ModelCheckpoint(
            MODEL_OUTPUT,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    print(f"\n=== Starting Training for {EPOCHS} Epochs ===\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n=== Training Complete! ===")
    print(f"Best model saved to: {MODEL_OUTPUT}")
    print(f"View training progress: tensorboard --logdir={LOG_DIR}")

    return model, history

if __name__ == '__main__':
    model, history = train()
```

---

## Appendix B: Evaluation Script

```python
#!/usr/bin/env python3
"""
DeNAT Model Evaluation Script
Calculate performance metrics matching manuscript benchmarking
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_dir):
    """Evaluate trained model on test set"""

    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}\n")

    # Load test data (no augmentation)
    test_ds = create_dataset(test_dir, batch_size=32, is_training=False)

    # Get predictions
    print("Generating predictions...")
    all_labels = []
    all_preds = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds[:, 1])  # Probability of positive class

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Apply threshold
    threshold = 0.5
    binary_preds = (all_preds > threshold).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0

    # Print results
    print("=== Performance Metrics ===\n")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}\n")

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"F-Score:              {f_score:.4f}")
    print(f"False Discovery Rate: {fdr:.4f}\n")

    # Target metrics from manuscript
    print("=== Target Metrics (from manuscript) ===\n")
    print("Sensitivity: 1.0")
    print("Precision:   0.948")
    print("F-Score:     0.973")
    print("FDR:         0.052\n")

    # Confusion matrix visualization
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'sensitivity': sensitivity,
        'precision': precision,
        'f_score': f_score,
        'fdr': fdr
    }

if __name__ == '__main__':
    metrics = evaluate_model(
        model_path='models/denat_pyramidotomy_final.h5',
        test_dir='DeNAT_Training_Data/test_images'
    )
```

---

## Appendix C: Spatial Binning Parameters

### Standard Distance Bins (Pyramidotomy)

Matching established pyramidotomy literature [Starkey et al., 2005; Wang et al., 2018; Venkatesh et al., 2021]:

| Bin | Distance from Midline | Biological Significance |
|-----|----------------------|-------------------------|
| Bin 1 | 0-200 μm | Immediate cross-midline sprouting |
| Bin 2 | 200-400 μm | Near-field plasticity |
| Bin 3 | 400-600 μm | Mid-field reorganization |
| Bin 4 | 600-800 μm | Far-field sprouting |
| Bin 5 | >800 μm | Long-distance axonal growth |

### Counting Line Specifications

- **Width**: 10 μm
- **Orientation**: Perpendicular to midline
- **Spacing**: Continuous coverage across bins

### Customizing Bins in GUI

```javascript
// Example: Custom bin configuration
const customBins = [
    {min: 0, max: 150},    // Narrower near-midline bin
    {min: 150, max: 300},
    {min: 300, max: 500},
    {min: 500, max: 1000}  // Wider far-field bin
];
```

---

## Appendix D: Fiber Index Normalization

### Purpose

Normalize for inter-animal variability in AAV tracing efficiency.

### Calculation

**Fiber Index** = (Number of crossing axons in spinal cord) / (Number of labeled axons in medullary pyramid)

### Protocol

1. **Image Medullary Pyramids**:
   - 50 μm sections of medulla
   - Zeiss ApoTome.2 microscope
   - Maximum intensity projections

2. **Count Pyramid Axons**:
   - Manual count of all AAV-labeled fibers in pyramid
   - Use consistent counting frame across animals

3. **Count Crossing Axons**:
   - Automated DeNAT count in cervical spinal cord
   - Sum across all distance bins

4. **Calculate Index**:
   ```python
   fiber_index = spinal_cord_count / pyramid_count
   ```

5. **Statistical Analysis**:
   - Use fiber index (not raw counts) for group comparisons
   - Accounts for tracing efficiency variability

---

## Appendix E: Quick Reference

### Key Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Image Size** | 4250 × 3350 px | Full-resolution confocal images |
| **Patch Size** | 256 × 256 px | Classification unit |
| **Stride** | 128 px | 50% overlap between patches |
| **Batch Size** | 32 | Training batch size |
| **Epochs** | 300 | Total training iterations |
| **Learning Rate** | 0.0001 | Adam optimizer LR |
| **Label Smoothing** | 0.1 | Smooth targets to 0.1/0.9 |
| **Dropout** | 0.2 | Regularization before output |
| **Augmentation Shift** | ±60 px | Random translation range |

### Performance Targets

| Metric | Target | Achieved (Manuscript) |
|--------|--------|----------------------|
| Sensitivity | ≥0.95 | 0.92 |
| Precision | ≥0.90 | 0.948 |
| F-Score | ≥0.90 | 0.973 |
| FDR | ≤0.10 | 0.052 |
| Correlation (R) | ≥0.95 | 0.9991 |

### File Locations

- **Training Code**: `model/training.py`
- **Architecture**: `model/architecture.py`
- **Data Pipeline**: `model/data.py`
- **Web App**: https://neuriteanalysis.netlify.app
- **GitHub**: https://github.com/mano2991/Neurite-Analysis-Tool

---

## Support and Resources

### Documentation

- **Manuscript**: Kumaran et al., "Deep Neurite Analysis Tool (DeNAT): A machine-learning framework for high-sensitivity automated neurite outgrowth measurement" (BMC Bioinformatics)
- **Video Tutorial**: Embedded in web application and GitHub repository
- **"How to Use" Guide**: Available on web application

### Data Privacy

- All image processing occurs locally in your browser
- No data uploaded to external servers
- Images not stored or retained after session ends
- Full data ownership retained by user

### Licensing

- **License**: MIT License
- **Free for**: Academic and commercial use
- **Permissions**: Use, modification, distribution
- **Attribution**: Please cite the manuscript in publications

### Citation

```
Kumaran M, Narayan PS A, Sahu Y, Banerjee S, Menon AS, Soni S, Venkatesh I.
Deep Neurite Analysis Tool (DeNAT): A machine-learning framework for
high-sensitivity automated neurite outgrowth measurement.
```

### Contact

- **Corresponding Author**: Ishwariya Venkatesh (ishwariya@ccmb.res.in)
- **Institution**: CSIR-Centre for Cellular and Molecular Biology (CCMB), Hyderabad, India

---

## Funding Acknowledgments

This research was supported by:
- Council of Scientific and Industrial Research (CSIR) (HCP53201)
- Department of Biotechnology (DBT) (BT/PR51467/MED/122/358/2024)
- Science and Engineering Research Board (SERB) (SRG00116)

---

**Authors**: DeNAT Development Team
