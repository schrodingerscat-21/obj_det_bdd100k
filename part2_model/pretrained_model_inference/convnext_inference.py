#!/usr/bin/env python3
"""
BDD100K ConvNeXt Model Inference Script
=====================================

This script runs inference on the BDD100K validation set using a pre-trained ConvNeXt model.
Supports both single image inference and batch processing of the validation set.

Requirements:
- torch, torchvision, mmdet, mmcv
- BDD100K dataset in proper format
- Pre-trained model checkpoint

Usage:
    python inference_script.py --model_path /path/to/model.pth --data_root /path/to/bdd100k --config-path /path/to/config.py --output_dir ./inference_results --single_image /path/to/image.jpg
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# MMDetection imports (assuming MMDet framework)
try:
    from mmdet.apis import init_detector, inference_detector
    from mmdet.core import get_classes
    from mmcv import Config
    MMDET_AVAILABLE = True
except ImportError:
    print("MMDetection not available. Using alternative approach.")
    MMDET_AVAILABLE = False

# BDD100K class names
BDD100K_CLASSES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

# Class colors for visualization
CLASS_COLORS = [
    (255, 0, 0),     # pedestrian - red
    (0, 255, 0),     # rider - green  
    (0, 0, 255),     # car - blue
    (255, 255, 0),   # truck - yellow
    (255, 0, 255),   # bus - magenta
    (0, 255, 255),   # train - cyan
    (128, 0, 128),   # motorcycle - purple
    (255, 165, 0),   # bicycle - orange
    (0, 128, 0),     # traffic light - dark green
    (128, 128, 128)  # traffic sign - gray
]

class BDD100KInference:
    """BDD100K Inference Class for ConvNeXt model"""
    
    def __init__(self, model_path, config_path=None, device='cuda'):
        """
        Initialize the inference engine
        
        Args:
            model_path (str): Path to the pre-trained model checkpoint
            config_path (str): Path to model config file (if using MMDet)
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.config = None
        
        # Load model
        if MMDET_AVAILABLE and config_path:
            self._load_mmdet_model(config_path, model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _load_mmdet_model(self, config_path, checkpoint_path):
        """Load model using MMDetection framework"""
        print(f"Loading MMDet model from {checkpoint_path}")
        self.config = Config.fromfile(config_path)
        self.model = init_detector(self.config, checkpoint_path, device=self.device)
        print("MMDet model loaded successfully")
    
    def _load_pytorch_model(self, model_path):
        """Load model using pure PyTorch (fallback method)"""
        print(f"Loading PyTorch model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model state dict (handle different checkpoint formats)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # TODO: Initialize your specific model architecture here
        # This is a placeholder - you'll need to match your exact model architecture
        print("Note: Using fallback PyTorch loading. You may need to modify the model architecture loading.")
        
        # Example model loading (adjust based on your model)
        # from your_model_file import ConvNeXtCascadeRCNN
        # self.model = ConvNeXtCascadeRCNN(num_classes=len(BDD100K_CLASSES))
        # self.model.load_state_dict(state_dict)
        # self.model.to(self.device)
        # self.model.eval()
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (adjust based on your training)
        # Common BDD100K sizes: (1280, 720) or (1024, 576)
        target_size = (1280, 720)
        image_resized = cv2.resize(image, target_size)
        
        # Normalize (ImageNet standard normalization)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        image_normalized = (image_resized - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, image
    
    def run_inference_single(self, image_path, conf_threshold=0.3):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to image
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            list: Detection results [bbox, score, class_id]
        """
        if MMDET_AVAILABLE and self.model:
            # Use MMDet inference
            result = inference_detector(self.model, image_path)
            return self._parse_mmdet_results(result, conf_threshold)
        else:
            # Use PyTorch inference
            image_tensor, original_image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            return self._parse_pytorch_results(predictions, conf_threshold)
    
    def _parse_mmdet_results(self, result, conf_threshold):
        """Parse MMDetection results"""
        detections = []
        
        for class_id, class_result in enumerate(result):
            if len(class_result) > 0:
                for detection in class_result:
                    bbox = detection[:4]  # x1, y1, x2, y2
                    score = detection[4]
                    
                    if score >= conf_threshold:
                        detections.append({
                            'bbox': bbox,
                            'score': score,
                            'class_id': class_id,
                            'class_name': BDD100K_CLASSES[class_id]
                        })
        
        return detections
    
    def _parse_pytorch_results(self, predictions, conf_threshold):
        """Parse PyTorch model results (placeholder)"""
        # This needs to be implemented based on your specific model output format
        # Different models have different output formats
        print("PyTorch result parsing needs to be implemented for your specific model")
        return []
    
    def visualize_results(self, image_path, detections, save_path=None):
        """
        Visualize detection results
        
        Args:
            image_path (str): Path to original image
            detections (list): Detection results
            save_path (str): Path to save visualization
        """
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image)
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            class_id = det['class_id']
            class_name = det['class_name']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Draw bounding box
            color = [c/255.0 for c in CLASS_COLORS[class_id]]
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {score:.2f}"
            ax.text(x1, y1-5, label, fontsize=10, color=color, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(f"Detections: {len(detections)} objects found")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def run_validation_set(self, data_root, output_dir, conf_threshold=0.3):
        """
        Run inference on entire validation set
        
        Args:
            data_root (str): Path to BDD100K dataset root
            output_dir (str): Directory to save results
            conf_threshold (float): Confidence threshold
        """
        val_images_dir = os.path.join(data_root, 'images', '10k', 'val')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all validation images
        image_paths = list(Path(val_images_dir).glob('*.jpg'))
        print(f"Found {len(image_paths)} validation images")
        
        results = {}
        
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}")
            
            # Run inference
            detections = self.run_inference_single(str(image_path), conf_threshold)
            
            # Store results
            image_name = image_path.name
            results[image_name] = detections
            
            # Save visualization for first 10 images
            if i < 10:
                viz_path = output_dir / f"viz_{image_name}"
                self.visualize_results(str(image_path), detections, str(viz_path))
        
        # Save all results to JSON
        results_path = output_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for img_name, dets in results.items():
                json_results[img_name] = []
                for det in dets:
                    json_det = {
                        'bbox': det['bbox'].tolist() if hasattr(det['bbox'], 'tolist') else list(det['bbox']),
                        'score': float(det['score']),
                        'class_id': int(det['class_id']),
                        'class_name': det['class_name']
                    }
                    json_results[img_name].append(json_det)
            
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        return results
    
    def evaluate_predictions(self, predictions, gt_annotations_path):
        """
        Evaluate predictions against ground truth (basic implementation)
        
        Args:
            predictions (dict): Prediction results
            gt_annotations_path (str): Path to ground truth annotations
        """
        # This is a simplified evaluation - for full COCO evaluation,
        # use pycocotools with proper format conversion
        
        print("Basic evaluation metrics:")
        
        total_predictions = 0
        class_counts = {cls: 0 for cls in BDD100K_CLASSES}
        
        for img_predictions in predictions.values():
            total_predictions += len(img_predictions)
            for pred in img_predictions:
                class_counts[pred['class_name']] += 1
        
        print(f"Total predictions: {total_predictions}")
        print("\nPredictions per class:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='BDD100K ConvNeXt Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config file (for MMDet)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to BDD100K dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save inference results')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = BDD100KInference(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    if args.single_image:
        # Run inference on single image
        print(f"Running inference on single image: {args.single_image}")
        detections = inference_engine.run_inference_single(
            args.single_image, args.conf_threshold
        )
        
        print(f"Found {len(detections)} detections")
        for det in detections:
            print(f"  {det['class_name']}: {det['score']:.3f}")
        
        # Visualize results
        inference_engine.visualize_results(args.single_image, detections)
    
    else:
        # Run inference on validation set
        print(f"Running inference on validation set")
        results = inference_engine.run_validation_set(
            args.data_root, args.output_dir, args.conf_threshold
        )
        
        # Basic evaluation
        inference_engine.evaluate_predictions(results, None)

if __name__ == "__main__":
    main()