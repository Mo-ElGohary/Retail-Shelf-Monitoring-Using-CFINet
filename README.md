# CFINet for SKU-110K Dataset

This repository contains the essential files to train and evaluate CFINet on the SKU-110K dataset for product detection.

## Files Included

- `cfinet.py` - CFINet model architecture implementation
- `dataset.py` - SKU-110K dataset loader (supports individual .txt annotation files)
- `train_sku110k.py` - Training script for SKU-110K dataset
- `inference_sku110k.py` - Inference/evaluation script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Structure

The dataset should be organized as follows with individual .txt annotation files:

```
/path/to/sku110k/
├── images/
│   ├── train/          # Training images (.jpg, .png, etc.)
│   ├── val/            # Validation images
│   └── test/           # Test images
└── labels/
    ├── train/          # Training annotations (.txt files)
    ├── val/            # Validation annotations
    └── test/           # Test annotations
```

### 3. Annotation Format

Each image should have a corresponding .txt file with the same name. The annotation file can be in one of these formats:

**YOLO Format (recommended):**
```
class_id x_center y_center width height
```
Example:
```
0 0.5 0.3 0.2 0.4
0 0.7 0.6 0.15 0.25
```

**Pascal VOC Format:**
```
x1 y1 x2 y2 class_id
```
Example:
```
100 150 200 300 0
350 200 450 350 0
```

**Simple Format:**
```
x1 y1 x2 y2
```
Example:
```
100 150 200 300
350 200 450 350
```

**Notes:**
- Coordinates can be normalized (0-1) or absolute pixel values
- All objects are treated as products (class_id = 0 or 1)
- If an image has no objects, the .txt file can be empty

## Training

### Basic Training Command

```bash
python train_sku110k.py \
    --data_root /path/to/sku110k \
    --output_dir ./outputs \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --img_size 1024
```

### Training Parameters

- `--data_root`: Path to SKU-110K dataset root directory
- `--output_dir`: Directory to save checkpoints and logs
- `--batch_size`: Training batch size (adjust based on GPU memory)
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Initial learning rate
- `--img_size`: Input image size (default: 512)
- `--num_workers`: Number of data loader workers (default: 4)
- `--resume`: Path to checkpoint to resume training from

### Example Training Commands

**Start new training:**
```bash
python train_sku110k.py --data_root /data/SKU110K --output_dir ./outputs --batch_size 16 --num_epochs 100
```

**Resume training from checkpoint:**
```bash
python train_sku110k.py --data_root /data/SKU110K --output_dir ./outputs --resume ./outputs/checkpoint_epoch_25.pth
```

## Evaluation/Inference

### Basic Evaluation Command

```bash
python inference_sku110k.py \
    --model_path ./outputs/best_model.pth \
    --data_root /path/to/sku110k \
    --split val \
    --output_dir ./inference_results
```

### Evaluation Parameters

- `--model_path`: Path to trained model checkpoint
- `--data_root`: Path to SKU-110K dataset root directory
- `--split`: Dataset split to evaluate on (train/val/test)
- `--output_dir`: Directory to save results
- `--confidence_threshold`: Confidence threshold for detections (default: 0.5)
- `--img_size`: Input image size (default: 512)
- `--num_samples`: Number of samples to evaluate (None for all)

### Example Evaluation Commands

**Evaluate on validation set:**
```bash
python inference_sku110k.py --model_path ./outputs/best_model.pth --data_root /data/SKU110K --split val
```

**Evaluate on test set with custom threshold:**
```bash
python inference_sku110k.py --model_path ./outputs/best_model.pth --data_root /data/SKU110K --split test --confidence_threshold 0.7
```

**Evaluate on subset of samples:**
```bash
python inference_sku110k.py --model_path ./outputs/best_model.pth --data_root /data/SKU110K --split val --num_samples 1000
```

## Output Files

### Training Outputs

- `outputs/checkpoint_epoch_X.pth` - Checkpoints for each epoch
- `outputs/best_model.pth` - Best model based on validation loss
- `training.log` - Training logs

### Evaluation Outputs

- `inference_results/inference_results_split_timestamp.json` - Detailed detection results
- `inference_results/summary_split_timestamp.json` - Summary statistics

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: RTX 3080 or better)
- **RAM**: At least 16GB system RAM
- **Storage**: ~50GB free space for dataset and outputs

## Performance Tips

1. **Batch Size**: Start with batch_size=8 and increase if GPU memory allows
2. **Image Size**: 512x512 is a good balance between accuracy and speed
3. **Data Loading**: Use multiple workers (--num_workers 4-8) for faster data loading
4. **Mixed Precision**: Consider using torch.cuda.amp for faster training on modern GPUs

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or image_size
2. **Slow Training**: Increase num_workers, use SSD storage
3. **Dataset Loading Errors**: Verify dataset structure and file paths
4. **Annotation Format**: Ensure .txt files follow the expected format

### Logs

Check `training.log` for detailed training progress and any error messages.

## Model Architecture

CFINet is a lightweight object detection model designed for product detection:
- Backbone: ResNet-18 with FPN
- Head: Classification and regression heads
- Output: Product bounding boxes and confidence scores

## Contact

For questions or issues, refer to the original CFINet paper and implementation. 
