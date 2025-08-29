import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

from cfinet import CFINet
from dataset import SKU110KDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        cls_scores, reg_deltas, proposals = model(images)
        
        # Calculate loss
        loss = criterion(cls_scores, reg_deltas, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / num_batches
    logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    return avg_loss

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Validation Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            cls_scores, reg_deltas, proposals = model(images)
            
            # Calculate loss
            loss = criterion(cls_scores, reg_deltas, targets)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Val Loss': f'{loss.item():.4f}',
                'Avg Val Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / num_batches
    logging.info(f'Validation Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, save_path)
    logging.info(f'Checkpoint saved: {save_path}')

def main():
    parser = argparse.ArgumentParser(description='Train CFINet on SKU-110K Dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to SKU-110K dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Input image size (recommended: 1024 for better detection of small objects)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use for training/validation (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create datasets
    logging.info('Creating datasets...')
    train_dataset = SKU110KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size
    )
    
    val_dataset = SKU110KDataset(
        root_dir=args.data_root,
        split='val',
        img_size=args.img_size
    )
    
    # Limit samples if specified
    if args.max_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.max_samples // 3, len(val_dataset))))  # Use fewer validation samples
        logging.info(f'Limited to {len(train_dataset)} training samples and {len(val_dataset)} validation samples')
    
    logging.info(f'Training samples: {len(train_dataset)}')
    logging.info(f'Validation samples: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logging.info('Creating CFINet model...')
    model = CFINet(num_classes=1, pretrained=True)
    model.to(device)
    
    # Create loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Simplified loss for demonstration
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        logging.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logging.info(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    logging.info('Starting training...')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            logging.info(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch+1} - LR: {current_lr:.6f}')
    
    logging.info('Training completed!')
    logging.info(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    main() 
