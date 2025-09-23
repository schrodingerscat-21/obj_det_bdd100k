import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def load_annotations(json_file_path):
    """Load and parse BDD100K annotation JSON file."""
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    
    # Create a mapping from image name to annotations
    annotation_dict = {}
    for annotation in annotations:
        image_name = annotation['name']
        annotation_dict[image_name] = annotation
    
    return annotation_dict

def get_category_color(category):
    """Return a color for each category type."""
    color_map = {
        'car': (0, 255, 0),           # Green
        'truck': (0, 255, 255),       # Yellow
        'bus': (255, 0, 255),         # Magenta
        'motorcycle': (255, 0, 0),    # Blue
        'bicycle': (0, 165, 255),     # Orange
        'pedestrian': (0, 0, 255),    # Red
        'rider': (255, 255, 0),       # Cyan
        'traffic sign': (128, 0, 128), # Purple
        'traffic light': (255, 192, 203), # Pink
        'train': (165, 42, 42),       # Brown
        'motor': (255, 20, 147),      # Deep Pink
        'trailer': (127, 255, 212),   # Aquamarine
        'drivable area': (50, 50, 50), # Dark Gray (usually not drawn as bbox)
        'lane': (100, 100, 100),      # Gray (usually not drawn as bbox)
    }
    return color_map.get(category, (255, 255, 255))  # Default to white

def draw_ground_truth_boxes(image, labels, draw_labels=True, box_thickness=2, font_scale=0.6):
    """Draw ground truth bounding boxes on the image."""
    for label in labels:
        category = label['category']
        
        # Skip drawing for drivable area and lane as they use poly2d, not box2d
        if category in ['drivable area', 'lane']:
            continue
            
        # Check if box2d exists (some labels might not have bounding boxes)
        if 'box2d' not in label:
            continue
            
        box = label['box2d']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # Get color for this category
        color = get_category_color(category)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
        
        # Draw label text if requested
        if draw_labels:
            label_text = category
            
            # Add traffic light color if available
            if category == 'traffic light' and 'attributes' in label:
                traffic_color = label['attributes'].get('trafficLightColor', 'none')
                if traffic_color != 'none':
                    label_text += f" ({traffic_color})"
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw text background
            cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(image, label_text, (x1, y1 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    
    return image

def find_matching_validation_image(base_name, validation_folder):
    """
    Find validation image with matching base name but potentially different extension.
    
    Args:
        base_name: Base filename without extension
        validation_folder: Path to validation images folder
        
    Returns:
        Path to matching validation image or None if not found
    """
    validation_path = Path(validation_folder)
    
    # Try common image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        candidate_path = validation_path / f"{base_name}{ext}"
        if candidate_path.exists():
            return candidate_path
    
    return None

def process_images(predicted_folder, validation_folder, annotation_file, output_folder, 
                  draw_labels=True, box_thickness=2, font_scale=0.6):
    """
    Process images and draw ground truth bounding boxes.
    
    Args:
        predicted_folder: Folder containing images with predicted boxes
        validation_folder: Folder containing original validation images
        annotation_file: JSON file with annotations
        output_folder: Folder to save ground truth visualizations
        draw_labels: Whether to draw category labels
        box_thickness: Thickness of bounding box lines
        font_scale: Scale of text labels
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(annotation_file)
    print(f"Loaded annotations for {len(annotations)} images")
    
    # Get list of predicted images
    predicted_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        predicted_images.extend(Path(predicted_folder).glob(ext))
    
    print(f"Found {len(predicted_images)} predicted images")
    
    processed_count = 0
    missing_annotations = 0
    missing_validation_images = 0
    extension_mismatches = 0
    
    for pred_image_path in predicted_images:
        pred_image_name = pred_image_path.name
        base_name = pred_image_path.stem  # filename without extension
        
        # First, try to find validation image with matching base name
        val_image_path = find_matching_validation_image(base_name, validation_folder)
        
        if val_image_path is None:
            print(f"Warning: No validation image found for base name: {base_name}")
            missing_validation_images += 1
            continue
        
        val_image_name = val_image_path.name
        
        # Check for extension mismatch
        if pred_image_name != val_image_name:
            extension_mismatches += 1
            if extension_mismatches <= 5:  # Only show first few mismatches to avoid spam
                print(f"Extension mismatch: {pred_image_name} -> {val_image_name}")
        
        # Check if annotation exists for the validation image name
        if val_image_name not in annotations:
            print(f"Warning: No annotation found for {val_image_name}")
            missing_annotations += 1
            continue
        
        # Load validation image
        image = cv2.imread(str(val_image_path))
        if image is None:
            print(f"Error: Could not load image {val_image_path}")
            continue
        
        # Get annotations for this image
        image_annotations = annotations[val_image_name]
        labels = image_annotations.get('labels', [])
        
        # Draw ground truth boxes
        image_with_gt = draw_ground_truth_boxes(
            image.copy(), labels, draw_labels, box_thickness, font_scale
        )
        
        # Save the image with ground truth boxes (using predicted image name for consistency)
        output_path = Path(output_folder) / f"gt_{pred_image_name}"
        cv2.imwrite(str(output_path), image_with_gt)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} images...")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Extension mismatches handled: {extension_mismatches}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Missing validation images: {missing_validation_images}")
    print(f"Ground truth images saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Draw ground truth bounding boxes on BDD100K validation images')
    parser.add_argument('--predicted_folder', type=str, required=True,
                       help='Folder containing images with predicted boxes')
    parser.add_argument('--validation_folder', type=str, required=True,
                       help='Folder containing original validation images')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='JSON file with BDD100K annotations')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Folder to save ground truth visualizations')
    parser.add_argument('--no_labels', action='store_true',
                       help='Do not draw category labels on boxes')
    parser.add_argument('--box_thickness', type=int, default=2,
                       help='Thickness of bounding box lines (default: 2)')
    parser.add_argument('--font_scale', type=float, default=0.6,
                       help='Scale of text labels (default: 0.6)')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.predicted_folder):
        print(f"Error: Predicted folder does not exist: {args.predicted_folder}")
        return
    
    if not os.path.exists(args.validation_folder):
        print(f"Error: Validation folder does not exist: {args.validation_folder}")
        return
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file does not exist: {args.annotation_file}")
        return
    
    # Process images
    process_images(
        predicted_folder=args.predicted_folder,
        validation_folder=args.validation_folder,
        annotation_file=args.annotation_file,
        output_folder=args.output_folder,
        draw_labels=not args.no_labels,
        box_thickness=args.box_thickness,
        font_scale=args.font_scale
    )

if __name__ == "__main__":
    main()