import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
from cfinet import CFINet
from dataset import SKU110KDataset, collate_fn

def train_cfinet(args):
    """Train CFINet on SKU-110K or other datasets"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CFINet(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Create datasets
    if args.dataset_type == 'sku110k':
        train_dataset = SKU110KDataset(
            root_dir=args.data_path,
            split='train',
            img_size=args.img_size
        )
        val_dataset = SKU110KDataset(
            root_dir=args.data_path,
            split='val',
            img_size=args.img_size
        )
    else:
        # For other datasets, you can modify this section
        print(f"Dataset type {args.dataset_type} not implemented yet")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Loss functions
    cls_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.SmoothL1Loss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in pbar:
            images = batch['images'].to(device)
            boxes = batch['boxes']
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            # Forward pass
            cls_scores, reg_deltas, proposals = model(images)
            
            # Calculate losses (simplified for demonstration)
            # In practice, you'd need proper anchor matching and loss calculation
            cls_loss = cls_criterion(cls_scores, torch.zeros_like(cls_scores))
            reg_loss = reg_criterion(reg_deltas, torch.zeros_like(reg_deltas))
            
            total_loss = cls_loss + reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'Loss': total_loss.item()})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for batch in pbar:
                images = batch['images'].to(device)
                boxes = batch['boxes']
                labels = batch['labels']
                
                # Forward pass
                cls_scores, reg_deltas, proposals = model(images)
                
                # Calculate losses
                cls_loss = cls_criterion(cls_scores, torch.zeros_like(cls_scores))
                reg_loss = reg_criterion(reg_deltas, torch.zeros_like(reg_deltas))
                
                total_loss = cls_loss + reg_loss
                val_loss += total_loss.item()
                pbar.set_postfix({'Loss': total_loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')

def main():
    parser = argparse.ArgumentParser(description='Train CFINet')
    parser.add_argument('--data_path', type=str, default='./data/SKU110K',
                        help='Path to dataset')
    parser.add_argument('--dataset_type', type=str, default='sku110k',
                        choices=['sku110k', 'pascal', 'coco'],
                        help='Type of dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr_step', type=int, default=30,
                        help='Learning rate step size')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_cfinet(args)

if __name__ == '__main__':
    main() 