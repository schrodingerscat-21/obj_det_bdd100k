#!/usr/bin/env python3
"""
YOLOv11 Training Script for BDD100K Dataset
Optimized for RTX 3060 with dataset-specific hyperparameters

Dataset Insights Addressed:
- Severe class imbalance (car:train = 5000:1)
- High occlusion rates (89% riders, 84% bikes)  
- Weather diversity (40% night scenes, foggy conditions)
- Dense scenes (18.4 objects/image average)
- RTX 3060 12GB VRAM constraint
"""

import os
import torch
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BDD100KTrainer:
    def __init__(self, data_path, model_size='m', device='0'):
        """
        Initialize BDD100K trainer with dataset-optimized configuration
        
        Args:
            data_path (str): Path to BDD100K dataset in YOLO format
            model_size (str): YOLOv11 model size ('n', 's', 'm', 'l', 'x')
            device (str): GPU device ID
        """
        self.data_path = Path(data_path)
        self.model_size = model_size
        self.device = device
        
        # RTX 3060 Memory Optimization
        self._optimize_gpu_memory()
        
        # Dataset-specific class information
        self.class_names = ['car', 'truck', 'bus', 'train', 'person', 'rider', 'bike', 'motor', 'traffic sign', 'traffic light']
        
        logger.info(f"Initializing YOLOv11{model_size} trainer for BDD100K")
        logger.info(f"Using device: {device}")
        logger.info("Note: Class imbalance will be handled by YOLO's built-in focal loss")
        
    def _optimize_gpu_memory(self):
        """Optimize GPU memory settings for RTX 3060"""
        # Enable memory optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Prevent memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("GPU memory optimization configured for RTX 3060")
    
    def create_dataset_yaml(self):
        """Create YAML configuration file optimized for BDD100K"""
        yaml_config = {
            'path': str(self.data_path),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': 10,  # Number of classes
            'names': self.class_names
        }
        
        yaml_path = self.data_path / 'bdd100k.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
            
        logger.info(f"Dataset YAML created at: {yaml_path}")
        return yaml_path
    
    def get_model_config(self):
        """Get model-specific configuration based on size and RTX 3060 constraints"""
        configs = {
            'n': {  # Nano - for edge deployment testing
                'batch': 32,
                'imgsz': 640,
                'epochs': 100,
                'patience': 15,
                'workers': 4
            },
            's': {  # Small - optimal for RTX 3060
                'batch': 32,
                'imgsz': 640, 
                'epochs': 100,
                'patience': 12,
                'workers': 4
            },
            'm': {  # Medium - max for RTX 3060
                'batch': 16,
                'imgsz': 640,
                'epochs': 80,
                'patience': 10,
                'workers': 4
            },
            'l': {  # Large - requires optimization
                'batch': 4,
                'imgsz': 640,
                'epochs': 60,
                'patience': 8,
                'workers': 2
            }
        }
        return configs.get(self.model_size, configs['s'])
    
    def get_training_hyperparameters(self):
        """
        Dataset-optimized hyperparameters based on BDD100K analysis
        
        Key optimizations:
        - Loss weights adjusted for class imbalance (car:train = 5000:1)
        - Enhanced augmentation for dense scenes (18.4 objects/image)
        - Color augmentation for weather variations (40% night scenes)
        - Occlusion handling for 89% rider, 84% bike occlusion
        
        Note: Using only valid YOLO arguments. Custom class weighting and 
        focal loss parameters are handled by YOLO's built-in mechanisms.
        """
        
        hyperparams = {
            # Optimizer settings - balanced for convergence and stability
            'lr0': 0.001,              # Initial learning rate
            'lrf': 0.00001,              # Final learning rate (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.05,    # Optimizer weight decay
            'warmup_epochs': 3.0,      # Warmup epochs
            'warmup_momentum': 0.8,    # Warmup initial momentum
            'warmup_bias_lr': 0.1,     # Warmup initial bias lr
            'cos_lr': True,
            
            # Loss function weights - addressing class imbalance
            'box': 7.5,               # Box loss gain (higher for precise localization)
            'cls': 0.5,               # Class loss gain (reduced due to imbalance)
            'dfl': 1.5,               # Distribution focal loss gain
            'label_smoothing': 0.0,    # Label smoothing (disabled for hard labels)
            
            # Data Augmentation - optimized for weather/occlusion robustness
            'hsv_h': 0.015,           # HSV-Hue augmentation (for lighting variations)
            'hsv_s': 0.7,             # HSV-Saturation augmentation (weather effects)
            'hsv_v': 0.4,             # HSV-Value augmentation (day/night)
            'degrees': 10.0,          # Rotation degree range (traffic scenarios)
            'translate': 0.1,         # Translation fraction (camera movement)
            'scale': 0.5,             # Scaling factor range (distance variation)
            'shear': 2.0,             # Shear degree range (perspective variation)
            'perspective': 0.0001,    # Perspective transform (camera angle)
            'flipud': 0.0,            # Vertical flip probability (disabled for driving)
            'fliplr': 0.5,            # Horizontal flip probability (lane switching)
            
            # Advanced Augmentation for Dense Scenes and Occlusion
            'mosaic': 0.8,            # Mosaic augmentation probability (dense scenes)
            'mixup': 0.15,            # MixUp augmentation probability (occlusion sim)
            'copy_paste': 0.3,        # Copy-paste augmentation (object interaction)
            
            # NMS settings optimized for dense traffic scenes
            'iou': 0.7,               # NMS IoU threshold (higher for dense scenes)
            'conf': 0.25,             # Confidence threshold (balanced)
            
            # Memory and Performance Optimization
            'amp': True,              # Automatic Mixed Precision (RTX 3060 optimization)
            'fraction': 1.0,          # Dataset fraction to use
            'profile': False,         # Profile ONNX and TensorRT speeds
            'freeze': None,           # Freeze layers (None = no freezing)
            'overlap_mask': True,     # Overlap masks for better segmentation
            'mask_ratio': 4,          # Mask downsample ratio
            'dropout': 0.0,           # Use dropout regularization
            'val': True,              # Validate/test during training
        }
        
        return hyperparams
    
    def setup_callbacks(self, save_dir):
        """Setup training callbacks for monitoring and optimization"""
        callbacks = {
            'on_train_start': self._on_train_start,
            'on_epoch_end': self._on_epoch_end,
            'on_train_end': self._on_train_end
        }
        return callbacks
    
    def _on_train_start(self, trainer):
        """Callback executed at training start"""
        logger.info("="*50)
        logger.info("BDD100K Training Started")
        logger.info(f"Model: YOLOv11{self.model_size}")
        logger.info(f"Dataset challenges addressed:")
        logger.info(f"  - Class imbalance: Built-in focal loss and loss weight adjustment")
        logger.info(f"  - High occlusion: Enhanced augmentation pipeline (mosaic, mixup)")
        logger.info(f"  - Weather diversity: Color space augmentation (HSV)")
        logger.info(f"  - Dense scenes: High IoU threshold and augmentation")
        logger.info("="*50)
    
    def _on_epoch_end(self, trainer):
        """Callback executed at end of each epoch"""
        epoch = trainer.epoch
        if epoch % 10 == 0:  # Log every 10 epochs
            logger.info(f"Epoch {epoch}: Loss={trainer.loss:.4f}")
            
            # Memory monitoring for RTX 3060
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
    
    def _on_train_end(self, trainer):
        """Callback executed at training completion"""
        logger.info("="*50)
        logger.info("BDD100K Training Completed!")
        logger.info(f"Best mAP: {trainer.best_fitness:.4f}")
        logger.info(f"Model saved to: {trainer.save_dir}")
        logger.info("="*50)
    
    def train(self, resume=False, pretrained=True):
        """
        Train YOLOv11 model with BDD100K-optimized configuration
        
        Args:
            resume (bool): Resume training from last checkpoint
            pretrained (bool): Use COCO pre-trained weights
        """
        try:
            # Create dataset configuration
            # yaml_path = self.create_dataset_yaml()
            yaml_path = self.data_path / 'data.yaml'
            
            # Initialize model
            model_name = f'yolo11{self.model_size}.pt' if pretrained else f'yolo11{self.model_size}.yaml'
            model = YOLO(model_name)
            
            logger.info(f"Model initialized: {model_name}")
            
            # Get configuration
            model_config = self.get_model_config()
            hyperparams = self.get_training_hyperparameters()
            
            # Merge configurations
            train_config = {
                **model_config,
                **hyperparams,
                'data': str(yaml_path),
                'device': self.device,
                'project': 'bdd100k_training',
                'name': f'yolo11{self.model_size}_bdd100k',
                'resume': resume,
                'save': True,
                'save_period': 1,  # Save checkpoint every 1 epochs
                # 'cache': True,      # Cache images for faster training
                'rect': True,       # Rectangular training
                'cos_lr': True,     # Cosine LR scheduler
                'close_mosaic': 10, # Disable mosaic last 10 epochs
                'resume': resume,
                'optimizer': 'AdamW',  # AdamW optimizer,

            }
            
            # Log training configuration
            logger.info("Training Configuration:")
            for key, value in train_config.items():
                logger.info(f"  {key}: {value}")
            
            # Start training
            logger.info("Starting training with BDD100K-optimized hyperparameters...")
            results = model.train(**train_config)
            
            # Validate model
            logger.info("Validating trained model...")
            val_results = model.val()
            
            # Export model for deployment
            logger.info("Exporting model for deployment...")
            model.export(format='onnx', dynamic=True, simplify=True)
            
            return results, val_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e
    
    def quick_test(self, subset_size=1000):
        """
        Quick training test with subset of data for validation
        
        Args:
            subset_size (int): Number of images to use for testing
        """
        logger.info(f"Running quick test with {subset_size} images...")
        
        # Create subset dataset
        self._create_subset_dataset(subset_size)
        
        # Reduced configuration for quick test
        quick_config = {
            'data': str(self.data_path / 'bdd100k_subset.yaml'),
            'epochs': 1,
            'batch': 8,
            'imgsz': 640,
            'device': self.device,
            'project': 'bdd100k_test',
            'name': f'yolo11{self.model_size}_quick_test',
            'patience': 0,
            'save': False,
            'plots': True,
            'val': True
        }
        
        # Initialize and train
        model = YOLO(f'yolo11{self.model_size}.pt')
        results = model.train(**quick_config)
        
        logger.info("Quick test completed successfully!")
        return results
    
    def _create_subset_dataset(self, subset_size):
        """Create subset dataset for quick testing"""
        # This is a placeholder - implement based on your dataset structure
        subset_yaml = {
            'path': str(self.data_path),
            'train': f'images/train_subset_{subset_size}',
            'val': f'images/val_subset_{int(subset_size*0.2)}',
            'nc': 10,
            'names': self.class_names
        }
        
        with open(self.data_path / 'bdd100k_subset.yaml', 'w') as f:
            yaml.dump(subset_yaml, f, default_flow_style=False)

def main():
    """Main training function with example usage"""
    
    # Configuration
    DATA_PATH = "/home/badri/bosch-project/yolo_data"  # Update this path
    MODEL_SIZE = 'm'  # Choose: 'n', 's', 'm', 'l'
    DEVICE = '0'      # GPU device ID
    
    # Initialize trainer
    trainer = BDD100KTrainer(
        data_path=DATA_PATH,
        model_size=MODEL_SIZE,
        device=DEVICE
    )
    
    # Option 1: Quick test with subset (recommended first)
    # print("Running quick test with 1000 images...")
    # trainer.quick_test(subset_size=1000)
    
    # Option 2: Full training (uncomment to run)
    print("Starting full training...")
    results, val_results = trainer.train(pretrained=True)
    print(f"Training completed. Best mAP: {val_results.box.map:.4f}")

if __name__ == "__main__":
    main()