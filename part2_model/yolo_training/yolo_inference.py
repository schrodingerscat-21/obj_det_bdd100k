#!/usr/bin/env python3
"""
YOLOv11 Inference Script for BDD100K Dataset
Performs object detection and draws bounding boxes on images

Compatible with models trained using the BDD100K training script.
Supports single images, batch processing, and video inference.
"""

import os
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import Union, List, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BDD100KInference:
	def __init__(self, model_path: str, device: str = '0', conf_threshold: float = 0.25, iou_threshold: float = 0.7):
		"""
		Initialize BDD100K inference engine

		Args:
			model_path (str): Path to trained YOLO model (.pt file)
			device (str): Device to run inference on ('0' for GPU, 'cpu' for CPU)
			conf_threshold (float): Confidence threshold for detections (0.0-1.0)
			iou_threshold (float): IoU threshold for NMS (0.0-1.0)
		"""
		self.model_path = Path(model_path)
		self.device = device
		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold

		# BDD100K class names (same as training script)
		self.class_names = [
			'car', 'truck', 'bus', 'train', 'person',
			'rider', 'bike', 'motor', 'traffic sign', 'traffic light'
		]

		# Class colors for visualization (BGR format)
		self.class_colors = {
			'car': (0, 255, 0),          # Green
			'truck': (255, 0, 0),        # Blue
			'bus': (0, 0, 255),          # Red
			'train': (255, 255, 0),      # Cyan
			'person': (255, 0, 255),     # Magenta
			'rider': (0, 255, 255),      # Yellow
			'bike': (128, 0, 128),       # Purple
			'motor': (255, 165, 0),      # Orange
			'traffic sign': (0, 128, 255), # Orange-Blue
			'traffic light': (255, 192, 203) # Pink
		}

		# Load model
		self._load_model()

	def _load_model(self):
		"""Load YOLO model and configure inference settings"""
		try:
			if not self.model_path.exists():
				logger.error(f"Model file not found: {self.model_path}")
				raise FileNotFoundError(f"Model file not found: {self.model_path}")

			logger.info(f"Loading model from: {self.model_path}")
			self.model = YOLO(str(self.model_path))

			# Configure model settings
			self.model.to(self.device)

			logger.info(f"Model loaded successfully on device: {self.device}")
			logger.info(f"Model input size: {getattr(self.model.model, 'imgsz', 640)}")
			logger.info(f"Confidence threshold: {self.conf_threshold}")
			logger.info(f"IoU threshold: {self.iou_threshold}")

		except Exception as e:
			logger.error(f"Failed to load model: {str(e)}")
			raise e

	def predict_image(self, image_path: str, save_path: str = None, show: bool = True) -> np.ndarray:
		"""
		Run inference on a single image

		Args:
			image_path (str): Path to input image
			save_path (str, optional): Path to save annotated image
			show (bool): Whether to display the image

		Returns:
			np.ndarray: Annotated image array
		"""
		try:
			# Load image
			image_path = Path(image_path)
			if not image_path.exists():
				raise FileNotFoundError(f"Image not found: {image_path}")

			logger.info(f"Processing image: {image_path}")

			# Run inference
			start_time = time.time()
			results = self.model(
				str(image_path),
				conf=self.conf_threshold,
				iou=self.iou_threshold,
				device=self.device,
				verbose=False
			)
			inference_time = time.time() - start_time

			# Load original image for annotation
			image = cv2.imread(str(image_path))
			if image is None:
				raise ValueError(f"Could not load image: {image_path}")

			# Draw bounding boxes
			annotated_image = self._draw_bounding_boxes(image, results[0])

			# Add inference time and stats to image
			annotated_image = self._add_info_text(annotated_image, results[0], inference_time)

			# Save annotated image
			if save_path:
				save_path = Path(save_path)
				save_path.parent.mkdir(parents=True, exist_ok=True)
				cv2.imwrite(str(save_path), annotated_image)
				logger.info(f"Annotated image saved to: {save_path}")

			# Display image
			if show:
				self._display_image(annotated_image, f"Detection Results - {image_path.name}")

			return annotated_image

		except Exception as e:
			logger.error(f"Error processing image: {str(e)}")
			raise e

	def predict_batch(self, image_dir: str, output_dir: str = None, pattern: str = "*.jpg") -> List[np.ndarray]:
		"""
		Run inference on multiple images in a directory

		Args:
			image_dir (str): Directory containing input images
			output_dir (str, optional): Directory to save annotated images
			pattern (str): File pattern to match (e.g., "*.jpg", "*.png")

		Returns:
			List[np.ndarray]: List of annotated image arrays
		"""
		image_dir = Path(image_dir)
		if not image_dir.exists():
			raise FileNotFoundError(f"Directory not found: {image_dir}")

		# Find all matching images
		image_files = list(image_dir.glob(pattern))
		if not image_files:
			logger.warning(f"No images found matching pattern '{pattern}' in {image_dir}")
			return []

		logger.info(f"Processing {len(image_files)} images from {image_dir}")

		annotated_images = []

		for i, image_file in enumerate(image_files, 1):
			logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")

			# Set output path if directory specified
			save_path = None
			if output_dir:
				output_dir = Path(output_dir)
				output_dir.mkdir(parents=True, exist_ok=True)
				save_path = output_dir / f"detected_{image_file.name}"

			# Process image
			try:
				annotated_image = self.predict_image(
					str(image_file),
					save_path=str(save_path) if save_path else None,
					show=False
				)
				annotated_images.append(annotated_image)
			except Exception as e:
				logger.error(f"Failed to process {image_file.name}: {str(e)}")

		logger.info(f"Batch processing completed. {len(annotated_images)} images processed successfully.")
		return annotated_images

	def predict_video(self, video_path: str, output_path: str = None, show: bool = True):
		"""
		Run inference on video frames

		Args:
			video_path (str): Path to input video
			output_path (str, optional): Path to save annotated video
			show (bool): Whether to display video during processing
		"""
		try:
			video_path = Path(video_path)
			if not video_path.exists():
				raise FileNotFoundError(f"Video not found: {video_path}")

			# Open video capture
			cap = cv2.VideoCapture(str(video_path))

			# Get video properties
			fps = int(cap.get(cv2.CAP_PROP_FPS))
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

			logger.info(f"Video info - FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")

			# Initialize video writer if output specified
			out = None
			if output_path:
				output_path = Path(output_path)
				output_path.parent.mkdir(parents=True, exist_ok=True)
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
				logger.info(f"Output video will be saved to: {output_path}")

			frame_count = 0
			start_time = time.time()

			while True:
				ret, frame = cap.read()
				if not ret:
					break

				frame_count += 1

				# Run inference on frame
				results = self.model(
					frame,
					conf=self.conf_threshold,
					iou=self.iou_threshold,
					device=self.device,
					verbose=False
				)

				# Draw bounding boxes
				annotated_frame = self._draw_bounding_boxes(frame, results[0])

				# Add frame info
				cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
						   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

				# Write frame to output video
				if out:
					out.write(annotated_frame)

				# Display frame
				if show:
					cv2.imshow('Video Detection', annotated_frame)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						logger.info("Video processing stopped by user")
						break

				# Progress update
				if frame_count % (fps * 5) == 0:  # Every 5 seconds
					elapsed = time.time() - start_time
					progress = frame_count / total_frames * 100
					logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {elapsed:.1f}s elapsed")

			# Cleanup
			cap.release()
			if out:
				out.release()
			if show:
				cv2.destroyAllWindows()

			processing_time = time.time() - start_time
			logger.info(f"Video processing completed in {processing_time:.2f}s")
			logger.info(f"Average FPS: {frame_count / processing_time:.2f}")

		except Exception as e:
			logger.error(f"Error processing video: {str(e)}")
			raise e

	def _draw_bounding_boxes(self, image: np.ndarray, results) -> np.ndarray:
		"""
		Draw bounding boxes and labels on image

		Args:
			image (np.ndarray): Input image
			results: YOLO detection results

		Returns:
			np.ndarray: Annotated image
		"""
		annotated_image = image.copy()

		if results.boxes is not None and len(results.boxes) > 0:
			boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
			confidences = results.boxes.conf.cpu().numpy()
			class_ids = results.boxes.cls.cpu().numpy().astype(int)

			for box, confidence, class_id in zip(boxes, confidences, class_ids):
				x1, y1, x2, y2 = map(int, box)
				class_name = self.class_names[class_id]
				color = self.class_colors.get(class_name, (255, 255, 255))

				# Draw bounding box
				cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

				# Prepare label
				label = f"{class_name}: {confidence:.2f}"
				label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

				# Draw label background
				cv2.rectangle(
					annotated_image,
					(x1, y1 - label_size[1] - 10),
					(x1 + label_size[0], y1),
					color,
					-1
				)

				# Draw label text
				cv2.putText(
					annotated_image,
					label,
					(x1, y1 - 5),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(255, 255, 255),
					2
				)

		return annotated_image

	def _add_info_text(self, image: np.ndarray, results, inference_time: float) -> np.ndarray:
		"""Add inference statistics to image"""
		height, width = image.shape[:2]

		# Detection count
		num_detections = len(results.boxes) if results.boxes is not None else 0

		# Info text
		info_texts = [
			f"Detections: {num_detections}",
			f"Inference: {inference_time*1000:.1f}ms",
			f"Conf: {self.conf_threshold}, IoU: {self.iou_threshold}"
		]

		# Draw info background
		text_height = 30
		bg_height = len(info_texts) * text_height + 10
		cv2.rectangle(image, (width - 300, 0), (width, bg_height), (0, 0, 0), -1)

		# Draw info text
		for i, text in enumerate(info_texts):
			y_pos = 25 + i * text_height
			cv2.putText(image, text, (width - 290, y_pos),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

		return image

	def _display_image(self, image: np.ndarray, window_name: str = "Detection Results"):
		"""Display image with OpenCV"""
		# Resize if image is too large
		height, width = image.shape[:2]
		max_size = 1200

		if max(height, width) > max_size:
			scale = max_size / max(height, width)
			new_width = int(width * scale)
			new_height = int(height * scale)
			image = cv2.resize(image, (new_width, new_height))

		cv2.imshow(window_name, image)
		logger.info("Press any key to continue...")
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def get_model_info(self):
		"""Get information about the loaded model"""
		info = {
			'model_path': str(self.model_path),
			'device': self.device,
			'conf_threshold': self.conf_threshold,
			'iou_threshold': self.iou_threshold,
			'num_classes': len(self.class_names),
			'class_names': self.class_names
		}
		return info

def main():
	"""Main function with command-line interface"""
	parser = argparse.ArgumentParser(description="YOLOv11 BDD100K Inference Script")
	parser.add_argument("--model", "-m", required=True, help="Path to trained YOLO model (.pt file)")
	parser.add_argument("--source", "-s", required=True, help="Path to image, directory, or video")
	parser.add_argument("--output", "-o", help="Output directory for annotated results")
	parser.add_argument("--device", "-d", default="0", help="Device: '0' for GPU, 'cpu' for CPU")
	parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
	parser.add_argument("--iou", "-i", type=float, default=0.7, help="IoU threshold for NMS (0.0-1.0)")
	parser.add_argument("--show", action="store_true", help="Display results")
	parser.add_argument("--pattern", default="*.jpg", help="File pattern for batch processing")

	args = parser.parse_args()

	# Initialize inference engine
	logger.info("Initializing BDD100K Inference Engine...")
	inference = BDD100KInference(
		model_path=args.model,
		device=args.device,
		conf_threshold=args.conf,
		iou_threshold=args.iou
	)

	# Print model info
	logger.info("Model Information:")
	for key, value in inference.get_model_info().items():
		logger.info(f"  {key}: {value}")

	source_path = Path(args.source)

	try:
		if source_path.is_file():
			# Check if it's a video file
			video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
			if source_path.suffix.lower() in video_extensions:
				logger.info("Processing video...")
				output_path = None
				if args.output:
					output_path = Path(args.output) / f"detected_{source_path.name}"
				inference.predict_video(str(source_path), str(output_path) if output_path else None, args.show)
			else:
				logger.info("Processing single image...")
				output_path = None
				if args.output:
					output_path = Path(args.output) / f"detected_{source_path.name}"
				inference.predict_image(str(source_path), str(output_path) if output_path else None, args.show)

		elif source_path.is_dir():
			logger.info("Processing image directory...")
			inference.predict_batch(str(source_path), args.output, args.pattern)

		else:
			logger.error(f"Source path not found: {source_path}")

	except Exception as e:
		logger.error(f"Inference failed: {str(e)}")
		return 1

	logger.info("Inference completed successfully!")
	return 0

# Example usage functions
def example_single_image():
	"""Example: Inference on single image"""
	# Initialize inference engine
	inference = BDD100KInference(
		model_path="path/to/your/trained_model.pt",  # Update this path
		device="0",
		conf_threshold=0.3,
		iou_threshold=0.7
	)

	# Run inference
	result = inference.predict_image(
		image_path="path/to/your/test_image.jpg",  # Update this path
		save_path="path/to/output/detected_image.jpg",  # Update this path
		show=True
	)

def example_batch_processing():
	"""Example: Batch processing of images"""
	inference = BDD100KInference(
		model_path="path/to/your/trained_model.pt",  # Update this path
		device="0"
	)

	results = inference.predict_batch(
		image_dir="path/to/your/test_images/",  # Update this path
		output_dir="path/to/output_directory/",  # Update this path
		pattern="*.jpg"
	)
	print(f"Processed {len(results)} images")

def example_video_processing():
	"""Example: Video processing"""
	inference = BDD100KInference(
		model_path="path/to/your/trained_model.pt",  # Update this path
		device="0"
	)

	inference.predict_video(
		video_path="path/to/your/test_video.mp4",  # Update this path
		output_path="path/to/output/detected_video.mp4",  # Update this path
		show=True
	)

if __name__ == "__main__":
	import sys
	sys.exit(main())