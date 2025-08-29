import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SKU110KDataset(Dataset):
    """
    SKU-110K Dataset loader for CFINet training
    SKU-110K is a large-scale dataset for retail product detection
    Supports individual .txt annotation files for each image
    """
    
    def __init__(self, root_dir, split='train', transform=None, img_size=512):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Define transforms
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(height=img_size, width=img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_size, width=img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform
            
        # Load dataset annotations
        self.images, self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """Load SKU-110K annotations from individual .txt files"""
        images = []
        annotations = []
        
        # Directory structure: root_dir/images/split/ and root_dir/labels/split/
        img_dir = os.path.join(self.root_dir, 'images', self.split)
        label_dir = os.path.join(self.root_dir, 'labels', self.split)
        
        # Check if directories exist
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
        
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory not found: {label_dir}")
            print("Will create empty annotations for all images")
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(img_dir) 
                      if f.lower().endswith(image_extensions)]
        
        print(f"Found {len(image_files)} images in {img_dir}")
        
        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)
            
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            images.append(img_path)
            
            # Load annotations from .txt file
            boxes = []
            labels = []
            
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:  # Skip empty lines
                                # Expected format: class_id x_center y_center width height
                                # or: x1 y1 x2 y2 class_id
                                parts = line.split()
                                if len(parts) == 5:
                                    if parts[0].isdigit():  # YOLO format: class_id x_center y_center width height
                                        class_id = int(parts[0])
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        
                                        # Convert to x1, y1, x2, y2 format
                                        x1 = x_center - width / 2
                                        y1 = y_center - height / 2
                                        x2 = x_center + width / 2
                                        y2 = y_center + height / 2
                                        
                                        boxes.append([x1, y1, x2, y2])
                                        labels.append(1)  # All objects are products
                                    else:  # Pascal VOC format: x1 y1 x2 y2 class_id
                                        x1 = float(parts[0])
                                        y1 = float(parts[1])
                                        x2 = float(parts[2])
                                        y2 = float(parts[3])
                                        class_id = int(parts[4])
                                        
                                        boxes.append([x1, y1, x2, y2])
                                        labels.append(1)  # All objects are products
                                elif len(parts) == 4:  # x1 y1 x2 y2 format
                                    x1 = float(parts[0])
                                    y1 = float(parts[1])
                                    x2 = float(parts[2])
                                    y2 = float(parts[3])
                                    
                                    boxes.append([x1, y1, x2, y2])
                                    labels.append(1)  # All objects are products
                except Exception as e:
                    print(f"Warning: Could not parse annotation file {label_path}: {e}")
                    boxes = []
                    labels = []
            else:
                # No annotation file found
                boxes = []
                labels = []
            
            annotations.append({
                'boxes': np.array(boxes, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64)
            })
        
        print(f"Loaded {len(images)} images with annotations")
        total_objects = sum(len(ann['boxes']) for ann in annotations)
        print(f"Total objects: {total_objects}")
        
        return images, annotations
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get annotations
        ann = self.annotations[idx]
        boxes = ann['boxes']
        labels = ann['labels']
        
        # Convert normalized coordinates to absolute if needed
        if len(boxes) > 0:
            # Check if coordinates are normalized (0-1) or absolute
            if np.max(boxes) <= 1.0:
                # Convert normalized to absolute coordinates
                boxes[:, [0, 2]] *= original_width
                boxes[:, [1, 3]] *= original_height
        
        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        else:
            transformed = self.transform(image=image, bboxes=[], labels=[])
            image = transformed['image']
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            labels = np.array([], dtype=np.int64)
        
        return {
            'image': image,
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels),
            'image_id': idx
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    images = []
    boxes = []
    labels = []
    image_ids = []
    
    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        labels.append(sample['labels'])
        image_ids.append(sample['image_id'])
    
    images = torch.stack(images, dim=0)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids
    }
