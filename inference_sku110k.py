import torch
import cv2
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import logging
from datetime import datetime

from cfinet import CFINet
from dataset import SKU110KDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path, device):
    """Load trained CFINet model"""
    model = CFINet(num_classes=1, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info(f'Model loaded from: {model_path}')
    logging.info(f'Training completed at epoch: {checkpoint["epoch"]}')
    return model

def preprocess_image(image_path, img_size=512):
    """Preprocess image for inference"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image, (img_size, img_size))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image

def detect_objects(model, image_tensor, device, confidence_threshold=0.5):
    """Perform object detection using CFINet"""
    with torch.no_grad():
        # Forward pass
        cls_scores, reg_deltas, proposals = model(image_tensor.to(device))
        
        # Convert to probabilities
        cls_probs = torch.sigmoid(cls_scores)
        
        # Get predictions above threshold
        predictions = []
        for i in range(cls_probs.shape[0]):  # batch dimension
            for j in range(cls_probs.shape[1]):  # class dimension
                for h in range(cls_probs.shape[2]):  # height
                    for w in range(cls_probs.shape[3]):  # width
                        confidence = cls_probs[i, j, h, w].item()
                        
                        if confidence > confidence_threshold:
                            # Get bounding box coordinates
                            # This is a simplified version - in practice you'd need proper anchor decoding
                            x1, y1, x2, y2 = reg_deltas[i, :, h, w].cpu().numpy()
                            
                            predictions.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': j
                            })
    
    return predictions

def evaluate_on_dataset(model, dataset, device, confidence_threshold=0.5):
    """Evaluate model on entire dataset"""
    model.eval()
    results = []
    
    logging.info(f'Evaluating on {len(dataset)} samples...')
    
    for i in tqdm(range(len(dataset)), desc='Evaluating'):
        sample = dataset[i]
        image_tensor = sample['image'].unsqueeze(0)
        
        # Perform detection
        predictions = detect_objects(model, image_tensor, device, confidence_threshold)
        
        results.append({
            'sample_id': i,
            'num_detections': len(predictions),
            'predictions': predictions
        })
    
    return results

def save_results(results, output_path):
    """Save evaluation results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f'Results saved to: {output_path}')

def main():
    parser = argparse.ArgumentParser(description='CFINet Inference on SKU-110K')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to SKU-110K dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Input image size (recommended: 1024 for better detection of small objects)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create dataset
    dataset = SKU110KDataset(
        root_dir=args.data_root,
        split=args.split,
        img_size=args.img_size
    )
    
    # Limit samples if specified
    if args.num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    
    logging.info(f'Evaluating on {len(dataset)} samples from {args.split} split')
    
    # Evaluate
    results = evaluate_on_dataset(model, dataset, device, args.confidence_threshold)
    
    # Calculate statistics
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results)
    
    logging.info(f'Evaluation completed!')
    logging.info(f'Total detections: {total_detections}')
    logging.info(f'Average detections per image: {avg_detections:.2f}')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.output_dir, f'inference_results_{args.split}_{timestamp}.json')
    save_results(results, results_path)
    
    # Save summary
    summary = {
        'model_path': args.model_path,
        'dataset_split': args.split,
        'num_samples': len(dataset),
        'confidence_threshold': args.confidence_threshold,
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'timestamp': timestamp
    }
    
    summary_path = os.path.join(args.output_dir, f'summary_{args.split}_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f'Summary saved to: {summary_path}')

if __name__ == '__main__':
    main() 