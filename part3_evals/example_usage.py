#!/usr/bin/env python3
"""
Example usage script for BDD100K ground truth visualization.

This script shows how to use the ground truth visualization function
both programmatically and via command line.
"""

import os
from bdd100k_gt_viz import process_images

def example_usage():
    """Example of how to use the ground truth visualization programmatically."""
    
    # Define your paths
    predicted_folder = "path/to/predicted_images"  # Folder with predicted images
    validation_folder = "path/to/bdd100k/images/100k/val"  # BDD100K validation images
    annotation_file = "path/to/bdd100k/labels/bdd100k_labels_images_val.json"  # Annotation file
    output_folder = "path/to/ground_truth_visualizations"  # Output folder
    
    # Process images
    process_images(
        predicted_folder=predicted_folder,
        validation_folder=validation_folder,
        annotation_file=annotation_file,
        output_folder=output_folder,
        draw_labels=True,        # Draw category labels
        box_thickness=2,         # Box line thickness
        font_scale=0.6          # Text size
    )

def command_line_examples():
    """Print command line usage examples."""
    
    print("Command Line Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic usage:")
    print("python bdd100k_gt_viz.py \\")
    print("    --predicted_folder ./predicted_images \\")
    print("    --validation_folder ./bdd100k/images/100k/val \\")
    print("    --annotation_file ./bdd100k/labels/bdd100k_labels_images_val.json \\")
    print("    --output_folder ./ground_truth_visualizations")
    
    print("\n2. Without text labels:")
    print("python bdd100k_gt_viz.py \\")
    print("    --predicted_folder ./predicted_images \\")
    print("    --validation_folder ./bdd100k/images/100k/val \\")
    print("    --annotation_file ./bdd100k/labels/bdd100k_labels_images_val.json \\")
    print("    --output_folder ./ground_truth_visualizations \\")
    print("    --no_labels")
    
    print("\n3. Custom visualization parameters:")
    print("python bdd100k_gt_viz.py \\")
    print("    --predicted_folder ./predicted_images \\")
    print("    --validation_folder ./bdd100k/images/100k/val \\")
    print("    --annotation_file ./bdd100k/labels/bdd100k_labels_images_val.json \\")
    print("    --output_folder ./ground_truth_visualizations \\")
    print("    --box_thickness 3 \\")
    print("    --font_scale 0.8")

if __name__ == "__main__":
    print("BDD100K Ground Truth Visualization - Usage Examples")
    print("=" * 60)
    
    command_line_examples()
    
    print("\n\nProgrammatic Usage:")
    print("=" * 20)
    print("Uncomment and modify the example_usage() function call below:")
    print("# example_usage()")