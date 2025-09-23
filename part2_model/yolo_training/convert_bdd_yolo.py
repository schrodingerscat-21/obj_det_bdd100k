import json
import os
from PIL import Image
import argparse
from pathlib import Path
import shutil

class BDD100KToYOLO:
    def __init__(self, bdd_root_path, yolo_output_path):
        self.bdd_root_path = Path(bdd_root_path)
        self.yolo_output_path = Path(yolo_output_path)
        
        # BDD100K class mapping - 10 main classes for object detection
        self.class_mapping = {
            'car': 0,
            'truck': 1, 
            'bus': 2,
            'train': 3,
            'person': 4,
            'rider': 5,
            'bike': 6,
            'motor': 7,
            'traffic sign': 8,
            'traffic light': 9
        }
        
        # Standard BDD100K image size
        self.img_width = 1280
        self.img_height = 720
        
    def setup_yolo_structure(self):
        """Create YOLO dataset folder structure"""
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.yolo_output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Create classes.txt file
        with open(self.yolo_output_path / 'classes.txt', 'w') as f:
            for class_name in sorted(self.class_mapping.keys(), key=lambda x: self.class_mapping[x]):
                f.write(f"{class_name}\n")
                
        # Create data.yaml file for YOLO training
        yaml_content = f"""path: {self.yolo_output_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(self.class_mapping)}
names: {list(self.class_mapping.keys())}
"""
        with open(self.yolo_output_path / 'data.yaml', 'w') as f:
            f.write(yaml_content)
    
    def get_image_dimensions(self, image_name, split):
        """Get image dimensions, either from actual image or use standard BDD100K size"""
        image_path = self.bdd_root_path / 'images' / split / image_name
        
        if image_path.exists():
            try:
                with Image.open(image_path) as img:
                    return img.size  # Returns (width, height)
            except Exception as e:
                print(f"Warning: Could not read image {image_name}: {e}")
                print(f"Using default BDD100K dimensions: {self.img_width}x{self.img_height}")
                return self.img_width, self.img_height
        else:
            print(f"Warning: Image {image_name} not found. Using default dimensions.")
            return self.img_width, self.img_height
    
    def convert_bbox_to_yolo(self, box2d, img_width, img_height):
        """Convert BDD100K bounding box to YOLO format"""
        x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
        
        # Calculate center coordinates and dimensions
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize coordinates
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        # Ensure coordinates are within [0, 1] range
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return center_x, center_y, width, height
    
    def process_annotation_file(self, json_file_path, split):
        """Process a single JSON annotation file"""
        print(f"Processing {json_file_path} for {split} split...")
        
        with open(json_file_path, 'r') as f:
            annotations = json.load(f)
        
        processed_images = 0
        total_objects = 0
        
        for img_annotation in annotations:
            image_name = img_annotation['name']
            img_width, img_height = self.get_image_dimensions(image_name, split)
            
            # Prepare YOLO annotation lines
            yolo_lines = []
            
            # Process each label in the image
            for label in img_annotation.get('labels', []):
                category = label['category']
                
                # Only process object detection categories (skip segmentation like 'drivable area', 'lane')
                if category in self.class_mapping and 'box2d' in label:
                    class_id = self.class_mapping[category]
                    
                    # Convert bounding box to YOLO format
                    center_x, center_y, width, height = self.convert_bbox_to_yolo(
                        label['box2d'], img_width, img_height
                    )
                    
                    # Format: class_id center_x center_y width height
                    yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)
                    total_objects += 1
            
            # Save YOLO annotation file
            if yolo_lines:  # Only create file if there are valid annotations
                txt_filename = os.path.splitext(image_name)[0] + '.txt'
                txt_path = self.yolo_output_path / 'labels' / split / txt_filename
                
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                # Copy corresponding image
                src_img_path = self.bdd_root_path / 'images' / split / image_name
                dst_img_path = self.yolo_output_path / 'images' / split / image_name
                
                if src_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)
                else:
                    print(f"Warning: Source image {src_img_path} not found")
                
                processed_images += 1
        
        print(f"Processed {processed_images} images with {total_objects} objects for {split} split")
        return processed_images, total_objects
    
    def convert(self):
        """Main conversion function"""
        print("Starting BDD100K to YOLO conversion...")
        print(f"Source: {self.bdd_root_path}")
        print(f"Output: {self.yolo_output_path}")
        
        # Setup YOLO folder structure
        self.setup_yolo_structure()
        
        total_images = 0
        total_objects = 0
        
        # Process annotation files
        for split in ['train', 'val']:
            json_file = self.bdd_root_path / 'labels' / f'bdd100k_labels_images_{split}.json'
            
            if json_file.exists():
                images, objects = self.process_annotation_file(json_file, split)
                total_images += images
                total_objects += objects
            else:
                print(f"Warning: Annotation file {json_file} not found")
        
        # Handle test set (if exists, but usually no annotations)
        test_img_dir = self.bdd_root_path / 'images' / 'test'
        if test_img_dir.exists():
            print("Copying test images (no annotations available)...")
            for img_file in test_img_dir.glob('*.jpg'):
                dst_path = self.yolo_output_path / 'images' / 'test' / img_file.name
                shutil.copy2(img_file, dst_path)
                # Create empty annotation file
                txt_file = (self.yolo_output_path / 'labels' / 'test' / 
                           f"{img_file.stem}.txt")
                txt_file.touch()
        
        print(f"\nConversion completed!")
        print(f"Total images processed: {total_images}")
        print(f"Total objects converted: {total_objects}")
        print(f"Classes: {len(self.class_mapping)}")
        print(f"Output saved to: {self.yolo_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert BDD100K dataset to YOLO format')
    parser.add_argument('--bdd_path', type=str, required=True,
                       help='Path to BDD100K dataset root directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to output YOLO dataset directory')
    
    args = parser.parse_args()
    
    # Validate input path
    bdd_path = Path(args.bdd_path)
    if not bdd_path.exists():
        print(f"Error: BDD100K path {bdd_path} does not exist")
        return
    
    # Check for required directories
    required_dirs = ['images', 'labels']
    for dir_name in required_dirs:
        if not (bdd_path / dir_name).exists():
            print(f"Error: Required directory {dir_name} not found in {bdd_path}")
            return
    
    # Create converter and run conversion
    converter = BDD100KToYOLO(args.bdd_path, args.output_path)
    converter.convert()

if __name__ == "__main__":
    main()

# Example usage:
# python bdd100k_to_yolo.py --bdd_path /path/to/bdd100k --output_path /path/to/yolo_dataset