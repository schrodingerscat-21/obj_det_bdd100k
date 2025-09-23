#!/usr/bin/env python3
"""
BDD100K Object Detection Dataset Analysis.

This module provides comprehensive analysis of the BDD100K object detection
dataset including cross-scenario analysis, sample image extraction, and
improved visualizations.
"""

import argparse
import json
import shutil
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BDD100KAnalyzer:
	"""Analyzer for BDD100K object detection dataset.

	This class provides comprehensive analysis capabilities for the BDD100K
	dataset including statistical analysis, visualization generation, and
	sample extraction.

	Attributes:
		DETECTION_CLASSES: List of 10 object detection classes.
		annotations_path: Path to the annotations JSON file.
		data: Loaded annotation data.
		stats: Computed statistics dictionary.
		df_annotations: DataFrame containing annotation-level data.
		df_images: DataFrame containing image-level data.
	"""

	# Define the 10 object detection classes
	DETECTION_CLASSES = [
		'car', 'truck', 'bus', 'train', 'person',
		'rider', 'bike', 'motor', 'traffic sign', 'traffic light'
	]

	def __init__(self, annotations_path: str) -> None:
		"""Initialize the analyzer.

		Args:
			annotations_path: Path to BDD100K annotations JSON file.
		"""
		self.annotations_path = Path(annotations_path)
		self.data = None
		self.stats = {}
		self.df_annotations = None
		self.df_images = None

	def load_annotations(self, sample_size: int = None) -> None:
		"""Efficiently load and parse annotations.

		Args:
			sample_size: Number of images to sample. If None, loads all data.
		"""
		print(f"Loading annotations from {self.annotations_path}...")

		with open(self.annotations_path, 'r', encoding='utf-8') as f:
			self.data = json.load(f)

		if sample_size:
			self.data = self.data[:sample_size]

		print(f"Loaded {len(self.data)} images")

	def parse_to_dataframes(self) -> None:
		"""Convert JSON data to pandas DataFrames for efficient analysis."""
		annotations_list = []
		images_list = []

		for img_idx, img_data in enumerate(self.data):
			# Image-level data with attributes
			img_info = self._extract_image_info(img_data, img_idx)
			images_list.append(img_info)

			# Object-level data with attributes
			for label in img_data['labels']:
				if label['category'] not in self.DETECTION_CLASSES:
					continue

				ann_info = self._extract_annotation_info(
					label, img_data, img_idx
				)
				annotations_list.append(ann_info)

		self.df_annotations = pd.DataFrame(annotations_list)
		self.df_images = pd.DataFrame(images_list)

		print(f"Created DataFrames: {len(self.df_annotations)} annotations, "
			  f"{len(self.df_images)} images")

	def _extract_image_info(self, img_data: Dict, img_idx: int) -> Dict:
		"""Extract image-level information.

		Args:
			img_data: Image data dictionary from JSON.
			img_idx: Image index.

		Returns:
			Dictionary containing image-level information.
		"""
		num_detection_objects = sum(
			1 for label in img_data['labels']
			if label['category'] in self.DETECTION_CLASSES
		)

		img_info = {
			'image_name': img_data['name'],
			'image_idx': img_idx,
			'weather': img_data['attributes'].get('weather', 'unknown'),
			'scene': img_data['attributes'].get('scene', 'unknown'),
			'timeofday': img_data['attributes'].get('timeofday', 'unknown'),
			'num_objects': len(img_data['labels']),
			'num_detection_objects': num_detection_objects
		}

		# Add binary flags for easier filtering
		img_info.update(self._add_binary_flags(img_info))

		return img_info

	def _add_binary_flags(self, img_info: Dict) -> Dict:
		"""Add binary flags for easier filtering.

		Args:
			img_info: Image information dictionary.

		Returns:
			Dictionary with additional binary flags.
		"""
		flags = {}

		# Scene flags
		flags['is_city'] = img_info['scene'] == 'city street'
		flags['is_highway'] = img_info['scene'] == 'highway'
		flags['is_residential'] = img_info['scene'] == 'residential'

		# Time flags
		flags['is_daytime'] = img_info['timeofday'] == 'daytime'
		flags['is_night'] = img_info['timeofday'] == 'night'
		flags['is_dawn_dusk'] = img_info['timeofday'] in [
			'dawn/dusk', 'dawn', 'dusk'
		]

		# Weather flags
		flags['is_clear'] = img_info['weather'] == 'clear'
		flags['is_rainy'] = img_info['weather'] == 'rainy'
		flags['is_snowy'] = img_info['weather'] == 'snowy'
		flags['is_foggy'] = img_info['weather'] == 'foggy'

		# Challenging conditions flags
		flags['is_challenging_weather'] = img_info['weather'] in [
			'rainy', 'snowy', 'foggy'
		]
		flags['is_challenging'] = (
			flags['is_challenging_weather'] or
			flags['is_night'] or
			flags['is_dawn_dusk']
		)

		return flags

	def _extract_annotation_info(self, label: Dict, img_data: Dict,
								img_idx: int) -> Dict:
		"""Extract annotation-level information.

		Args:
			label: Label dictionary from JSON.
			img_data: Image data dictionary.
			img_idx: Image index.

		Returns:
			Dictionary containing annotation-level information.
		"""
		box = label['box2d']
		width = box['x2'] - box['x1']
		height = box['y2'] - box['y1']
		area = width * height

		ann_info = {
			'image_name': img_data['name'],
			'image_idx': img_idx,
			'category': label['category'],
			'x1': box['x1'],
			'y1': box['y1'],
			'x2': box['x2'],
			'y2': box['y2'],
			'width': width,
			'height': height,
			'area': area,
			'aspect_ratio': width / height if height > 0 else 0,
			'relative_size': area / (1280 * 720),  # Standard BDD100K resolution
			'occluded': label['attributes'].get('occluded', False),
			'truncated': label['attributes'].get('truncated', False),
			'weather': img_data['attributes'].get('weather', 'unknown'),
			'scene': img_data['attributes'].get('scene', 'unknown'),
			'timeofday': img_data['attributes'].get('timeofday', 'unknown')
		}

		# Add size category
		ann_info['size_category'] = self._get_size_category(area)

		# Add traffic light color if applicable
		if label['category'] == 'traffic light':
			ann_info['trafficLightColor'] = label['attributes'].get(
				'trafficLightColor', 'none'
			)

		return ann_info

	def _get_size_category(self, area: float) -> str:
		"""Determine size category based on area.

		Args:
			area: Object area in pixels.

		Returns:
			Size category string.
		"""
		if area < 32 * 32:
			return 'tiny'
		elif area < 96 * 96:
			return 'small'
		elif area < 256 * 256:
			return 'medium'
		else:
			return 'large'

	def compute_comprehensive_statistics(self) -> Dict:
		"""Compute comprehensive statistics including cross-scenario analysis.

		Returns:
			Dictionary containing comprehensive statistics.
		"""
		stats = {}

		# Overall statistics
		stats['total_images'] = len(self.df_images)
		stats['total_annotations'] = len(self.df_annotations)
		stats['avg_objects_per_image'] = self.df_images[
			'num_detection_objects'
		].mean()
		stats['std_objects_per_image'] = self.df_images[
			'num_detection_objects'
		].std()

		# Per-class detailed statistics
		stats['class_stats'] = self._compute_class_statistics()

		# Cross-scenario analysis
		stats['cross_scenario_analysis'] = self._compute_cross_scenario_analysis()

		# Geographic/Environmental diversity
		stats['diversity_stats'] = self._compute_diversity_statistics()

		# City vs Non-city analysis
		stats['city_analysis'] = self._compute_city_analysis()

		# Day vs Night analysis
		stats['day_night_analysis'] = self._compute_day_night_analysis()

		return stats

	def _compute_class_statistics(self) -> Dict:
		"""Compute per-class detailed statistics.

		Returns:
			Dictionary containing class-wise statistics.
		"""
		class_stats = {}

		for cls in self.DETECTION_CLASSES:
			cls_df = self.df_annotations[
				self.df_annotations['category'] == cls
			]

			if len(cls_df) == 0:
				class_stats[cls] = {
					'count': 0,
					'images_with_class': 0,
					'percentage_of_total': 0
				}
				continue

			class_stats[cls] = {
				'count': len(cls_df),
				'images_with_class': cls_df['image_name'].nunique(),
				'percentage_of_total': (
					len(cls_df) / len(self.df_annotations)
				) * 100,
				'avg_per_image': len(cls_df) / len(self.df_images),
				'avg_width': cls_df['width'].mean(),
				'avg_height': cls_df['height'].mean(),
				'avg_area': cls_df['area'].mean(),
				'median_area': cls_df['area'].median(),
				'std_area': cls_df['area'].std(),
				'avg_aspect_ratio': cls_df['aspect_ratio'].mean(),

				# Occlusion and truncation detailed stats
				'occluded_count': cls_df['occluded'].sum(),
				'occluded_ratio': cls_df['occluded'].mean() * 100,
				'truncated_count': cls_df['truncated'].sum(),
				'truncated_ratio': cls_df['truncated'].mean() * 100,
				'both_occluded_truncated': (
					(cls_df['occluded'] & cls_df['truncated']).sum()
				),

				# Size distribution
				'size_distribution': cls_df['size_category'].value_counts().to_dict(),

				# Environmental distribution
				'weather_distribution': cls_df['weather'].value_counts().to_dict(),
				'scene_distribution': cls_df['scene'].value_counts().to_dict(),
				'timeofday_distribution': cls_df['timeofday'].value_counts().to_dict()
			}

		return class_stats

	def _compute_cross_scenario_analysis(self) -> Dict:
		"""Compute cross-scenario analysis.

		Returns:
			Dictionary containing cross-scenario analysis results.
		"""
		cross_scenario = {}

		# Weather + Time combinations
		weather_time_combo = self.df_images.groupby(
			['weather', 'timeofday']
		).size()
		cross_scenario['weather_time_combinations'] = {
			f"{k[0]}_{k[1]}": v for k, v in weather_time_combo.to_dict().items()
		}

		# Scene + Time combinations
		scene_time_combo = self.df_images.groupby(['scene', 'timeofday']).size()
		cross_scenario['scene_time_combinations'] = {
			f"{k[0]}_{k[1]}": v for k, v in scene_time_combo.to_dict().items()
		}

		# Challenging scenarios
		challenging_scenarios = {
			'rainy_night': len(self.df_images[
				(self.df_images['is_rainy']) & (self.df_images['is_night'])
			]),
			'snowy_night': len(self.df_images[
				(self.df_images['is_snowy']) & (self.df_images['is_night'])
			]),
			'foggy_dawn_dusk': len(self.df_images[
				(self.df_images['is_foggy']) & (self.df_images['is_dawn_dusk'])
			]),
			'challenging_weather_night': len(self.df_images[
				(self.df_images['is_challenging_weather']) &
				(self.df_images['is_night'])
			])
		}
		cross_scenario['challenging_scenarios'] = challenging_scenarios

		# Object distribution in challenging scenarios
		challenging_images = self.df_images[self.df_images['is_challenging']]
		normal_images = self.df_images[~self.df_images['is_challenging']]

		cross_scenario['object_distribution_challenging'] = {
			'avg_objects_challenging': challenging_images[
				'num_detection_objects'
			].mean(),
			'avg_objects_normal': normal_images[
				'num_detection_objects'
			].mean(),
			'total_challenging_images': len(challenging_images),
			'total_normal_images': len(normal_images)
		}

		return cross_scenario

	def _compute_diversity_statistics(self) -> Dict:
		"""Compute environmental diversity statistics.

		Returns:
			Dictionary containing diversity statistics.
		"""
		diversity_stats = {
			'weather_diversity': {
				weather: (count / len(self.df_images)) * 100
				for weather, count in self.df_images['weather'].value_counts().items()
			},
			'scene_diversity': {
				scene: (count / len(self.df_images)) * 100
				for scene, count in self.df_images['scene'].value_counts().items()
			},
			'timeofday_diversity': {
				time: (count / len(self.df_images)) * 100
				for time, count in self.df_images['timeofday'].value_counts().items()
			}
		}

		return diversity_stats

	def _compute_city_analysis(self) -> Dict:
		"""Compute city vs non-city analysis.

		Returns:
			Dictionary containing city analysis results.
		"""
		city_images = self.df_images[self.df_images['is_city']]
		non_city_images = self.df_images[~self.df_images['is_city']]

		return {
			'city_images': len(city_images),
			'non_city_images': len(non_city_images),
			'avg_objects_city': city_images['num_detection_objects'].mean(),
			'avg_objects_non_city': non_city_images[
				'num_detection_objects'
			].mean()
		}

	def _compute_day_night_analysis(self) -> Dict:
		"""Compute day vs night analysis.

		Returns:
			Dictionary containing day/night analysis results.
		"""
		day_images = self.df_images[self.df_images['is_daytime']]
		night_images = self.df_images[self.df_images['is_night']]

		return {
			'day_images': len(day_images),
			'night_images': len(night_images),
			'dawn_dusk_images': len(
				self.df_images[self.df_images['is_dawn_dusk']]
			),
			'avg_objects_day': (
				day_images['num_detection_objects'].mean()
				if len(day_images) > 0 else 0
			),
			'avg_objects_night': (
				night_images['num_detection_objects'].mean()
				if len(night_images) > 0 else 0
			)
		}

	def find_interesting_samples(self) -> Dict:
		"""Find interesting and unique samples for each analysis category.

		Returns:
			Dictionary containing interesting samples for various categories.
		"""
		samples = {}

		# 1. Extreme challenging scenarios
		challenging_scenarios = [
			(
				'rainy_night',
				(self.df_images['is_rainy']) & (self.df_images['is_night'])
			),
			(
				'snowy_night',
				(self.df_images['is_snowy']) & (self.df_images['is_night'])
			),
			(
				'foggy_night',
				(self.df_images['is_foggy']) & (self.df_images['is_night'])
			),
			(
				'foggy_dawn',
				(self.df_images['is_foggy']) & (self.df_images['is_dawn_dusk'])
			)
		]

		for scenario_name, condition in challenging_scenarios:
			scenario_images = self.df_images[condition]
			if len(scenario_images) > 0:
				# Get images with most objects in this scenario
				top_crowded = scenario_images.nlargest(3, 'num_detection_objects')
				samples[f'{scenario_name}_crowded'] = top_crowded[
					['image_name', 'num_detection_objects']
				].to_dict('records')

		# 2. Heavily occluded scenes
		occlusion_by_image = self.df_annotations.groupby('image_name').agg({
			'occluded': 'mean',
			'truncated': 'mean',
			'category': 'count'
		})

		heavily_occluded = occlusion_by_image.nlargest(5, 'occluded')
		samples['heavily_occluded'] = heavily_occluded.reset_index()[
			['image_name', 'occluded']
		].to_dict('records')

		heavily_truncated = occlusion_by_image.nlargest(5, 'truncated')
		samples['heavily_truncated'] = heavily_truncated.reset_index()[
			['image_name', 'truncated']
		].to_dict('records')

		# 3. Class-specific interesting samples
		for cls in self.DETECTION_CLASSES:
			cls_df = self.df_annotations[
				self.df_annotations['category'] == cls
			]
			if len(cls_df) == 0:
				continue

			cls_samples = {}

			# Largest and smallest
			cls_samples['largest'] = cls_df.nlargest(2, 'area')[
				['image_name', 'area']
			].to_dict('records')
			cls_samples['smallest'] = cls_df.nsmallest(2, 'area')[
				['image_name', 'area']
			].to_dict('records')

			# Most extreme aspect ratios
			cls_samples['widest'] = cls_df.nlargest(2, 'aspect_ratio')[
				['image_name', 'aspect_ratio']
			].to_dict('records')
			cls_samples['tallest'] = cls_df.nsmallest(2, 'aspect_ratio')[
				['image_name', 'aspect_ratio']
			].to_dict('records')

			samples[f'class_{cls}'] = cls_samples

		# 4. Dense scenes per category
		category_counts = self.df_annotations.groupby(
			['image_name', 'category']
		).size()

		for cls in ['car', 'person', 'traffic light', 'traffic sign']:
			if cls in category_counts.index.get_level_values('category'):
				dense = category_counts.xs(cls, level='category').nlargest(3)
				samples[f'dense_{cls}'] = [
					{'image_name': img, 'count': count}
					for img, count in dense.items()
				]

		return samples

	def create_visualizations(self, output_dir: Path) -> Path:
		"""Create visualizations with value labels.

		Args:
			output_dir: Directory to save visualizations.

		Returns:
			Path to the saved visualization file.
		"""
		# Create figure with subplots
		fig = plt.figure(figsize=(20, 16))
		gs = gridspec.GridSpec(4, 3, figure=fig)

		# 1. Class Distribution with values
		self._create_class_distribution_plot(fig, gs)

		# 2. Occlusion and Truncation Analysis
		self._create_occlusion_analysis_plot(fig, gs)

		# 3. Weather-Time Cross Analysis
		self._create_weather_time_plot(fig, gs)

		# 4. City vs Non-city Analysis
		self._create_city_analysis_plot(fig, gs)

		# 5. Day vs Night Analysis
		self._create_day_night_plot(fig, gs)

		# 6. Challenging Scenarios
		self._create_challenging_scenarios_plot(fig, gs)

		# 7. Size Distribution by Category
		self._create_size_distribution_plot(fig, gs)

		# 8. Object Count Distribution
		self._create_object_count_distribution_plot(fig, gs)

		# 9. Environmental Diversity Pie Chart
		self._create_environmental_diversity_plot(fig, gs)

		plt.suptitle(
			'BDD100K Object Detection: Comprehensive Analysis',
			fontsize=16, fontweight='bold', y=1.02
		)
		plt.tight_layout()

		# Save figure
		viz_path = output_dir / 'analysis_visualization.png'
		plt.savefig(viz_path, dpi=150, bbox_inches='tight')
		plt.close()

		return viz_path

	def _create_class_distribution_plot(self, fig, gs) -> None:
		"""Create class distribution plot."""
		ax1 = fig.add_subplot(gs[0, :])
		class_counts = self.df_annotations['category'].value_counts()
		bars = ax1.bar(
			range(len(class_counts)),
			class_counts.values,
			color='steelblue'
		)
		ax1.set_xticks(range(len(class_counts)))
		ax1.set_xticklabels(class_counts.index, rotation=45, ha='right')
		ax1.set_ylabel('Count (log scale)', fontsize=12)
		ax1.set_title('Object Class Distribution', fontsize=14, fontweight='bold')
		ax1.set_yscale('log')

		# Add value labels on bars
		for bar, value in zip(bars, class_counts.values):
			height = bar.get_height()
			ax1.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{value:,}',
				ha='center',
				va='bottom',
				fontsize=10
			)

	def _create_occlusion_analysis_plot(self, fig, gs) -> None:
		"""Create occlusion and truncation analysis plot."""
		ax2 = fig.add_subplot(gs[1, 0])
		occlusion_data = []
		truncation_data = []
		labels = []

		for cls in self.DETECTION_CLASSES:
			if cls in self.stats['class_stats']:
				occlusion_data.append(
					self.stats['class_stats'][cls]['occluded_ratio']
				)
				truncation_data.append(
					self.stats['class_stats'][cls]['truncated_ratio']
				)
				labels.append(cls)

		x = np.arange(len(labels))
		width = 0.35
		bars1 = ax2.bar(
			x - width / 2,
			occlusion_data,
			width,
			label='Occluded',
			color='coral'
		)
		bars2 = ax2.bar(
			x + width / 2,
			truncation_data,
			width,
			label='Truncated',
			color='skyblue'
		)

		ax2.set_xlabel('Class')
		ax2.set_ylabel('Percentage (%)')
		ax2.set_title('Occlusion & Truncation Rates by Class')
		ax2.set_xticks(x)
		ax2.set_xticklabels(labels, rotation=45, ha='right')
		ax2.legend()

		# Add value labels
		for bar in bars1:
			height = bar.get_height()
			ax2.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{height:.1f}%',
				ha='center',
				va='bottom',
				fontsize=8
			)
		for bar in bars2:
			height = bar.get_height()
			ax2.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{height:.1f}%',
				ha='center',
				va='bottom',
				fontsize=8
			)

	def _create_weather_time_plot(self, fig, gs) -> None:
		"""Create weather-time cross analysis plot."""
		ax3 = fig.add_subplot(gs[1, 1])
		weather_time = []

		for combo_key, count in self.stats['cross_scenario_analysis'][
			'weather_time_combinations'
		].items():
			weather, time = combo_key.split('_', 1)
			weather_time.append({
				'weather': weather,
				'time': time,
				'count': count
			})

		if weather_time:
			weather_time_df = pd.DataFrame(weather_time)
			pivot = weather_time_df.pivot(
				index='weather',
				columns='time',
				values='count'
			).fillna(0)

			im = ax3.imshow(pivot, cmap='YlOrRd', aspect='auto')
			ax3.set_xticks(range(len(pivot.columns)))
			ax3.set_yticks(range(len(pivot.index)))
			ax3.set_xticklabels(pivot.columns, rotation=45, ha='right')
			ax3.set_yticklabels(pivot.index)
			ax3.set_title('Weather × Time of Day Distribution')

			# Add text annotations
			for i in range(len(pivot.index)):
				for j in range(len(pivot.columns)):
					ax3.text(
						j, i,
						f'{int(pivot.iloc[i, j])}',
						ha="center",
						va="center",
						color="black",
						fontsize=9
					)

			plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

	def _create_city_analysis_plot(self, fig, gs) -> None:
		"""Create city vs non-city analysis plot."""
		ax4 = fig.add_subplot(gs[1, 2])
		city_data = self.stats['city_analysis']
		categories = ['City', 'Non-City']
		counts = [city_data['city_images'], city_data['non_city_images']]
		avg_objects = [
			city_data['avg_objects_city'],
			city_data['avg_objects_non_city']
		]

		x = np.arange(len(categories))
		width = 0.35

		ax4_twin = ax4.twinx()
		bars1 = ax4.bar(
			x - width / 2,
			counts,
			width,
			label='Image Count',
			color='lightblue'
		)
		bars2 = ax4_twin.bar(
			x + width / 2,
			avg_objects,
			width,
			label='Avg Objects',
			color='lightgreen'
		)

		ax4.set_xlabel('Scene Type')
		ax4.set_ylabel('Image Count', color='blue')
		ax4_twin.set_ylabel('Avg Objects per Image', color='green')
		ax4.set_title('City vs Non-City Analysis')
		ax4.set_xticks(x)
		ax4.set_xticklabels(categories)
		ax4.tick_params(axis='y', labelcolor='blue')
		ax4_twin.tick_params(axis='y', labelcolor='green')

		# Add value labels
		for bar in bars1:
			height = bar.get_height()
			ax4.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{int(height):,}',
				ha='center',
				va='bottom',
				fontsize=9
			)
		for bar in bars2:
			height = bar.get_height()
			ax4_twin.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{height:.1f}',
				ha='center',
				va='bottom',
				fontsize=9
			)

	def _create_day_night_plot(self, fig, gs) -> None:
		"""Create day vs night analysis plot."""
		ax5 = fig.add_subplot(gs[2, 0])
		day_night_data = self.stats['day_night_analysis']
		times = ['Day', 'Night', 'Dawn/Dusk']
		counts = [
			day_night_data['day_images'],
			day_night_data['night_images'],
			day_night_data['dawn_dusk_images']
		]

		bars = ax5.bar(times, counts, color=['gold', 'midnightblue', 'orange'])
		ax5.set_ylabel('Number of Images')
		ax5.set_title('Time of Day Distribution')

		# Add value labels and percentages
		total = sum(counts)
		for bar, count in zip(bars, counts):
			height = bar.get_height()
			ax5.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{count:,}\n({count/total*100:.1f}%)',
				ha='center',
				va='bottom',
				fontsize=10
			)

	def _create_challenging_scenarios_plot(self, fig, gs) -> None:
		"""Create challenging scenarios plot."""
		ax6 = fig.add_subplot(gs[2, 1])
		challenging = self.stats['cross_scenario_analysis']['challenging_scenarios']
		scenarios = list(challenging.keys())
		counts = list(challenging.values())

		bars = ax6.barh(range(len(scenarios)), counts, color='crimson')
		ax6.set_yticks(range(len(scenarios)))
		ax6.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])
		ax6.set_xlabel('Number of Images')
		ax6.set_title('Challenging Scenario Distribution')

		# Add value labels
		for bar, count in zip(bars, counts):
			width = bar.get_width()
			ax6.text(
				width,
				bar.get_y() + bar.get_height() / 2.,
				f'{count:,}',
				ha='left',
				va='center',
				fontsize=9
			)

	def _create_size_distribution_plot(self, fig, gs) -> None:
		"""Create size distribution plot."""
		ax7 = fig.add_subplot(gs[2, 2])
		size_data = {}

		for cls in ['car', 'person', 'truck', 'bus', 'traffic sign']:
			if cls in self.stats['class_stats']:
				size_dist = self.stats['class_stats'][cls].get(
					'size_distribution', {}
				)
				for size, count in size_dist.items():
					if size not in size_data:
						size_data[size] = {}
					size_data[size][cls] = count

		if size_data:
			size_df = pd.DataFrame(size_data).fillna(0).T
			size_df.plot(kind='bar', stacked=True, ax=ax7, colormap='viridis')
			ax7.set_xlabel('Size Category')
			ax7.set_ylabel('Count')
			ax7.set_title('Object Size Distribution by Top Classes')
			ax7.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
			plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0)

	def _create_object_count_distribution_plot(self, fig, gs) -> None:
		"""Create object count distribution plot."""
		ax8 = fig.add_subplot(gs[3, :2])
		ax8.hist(
			self.df_images['num_detection_objects'],
			bins=50,
			edgecolor='black',
			alpha=0.7
		)

		mean_val = self.df_images['num_detection_objects'].mean()
		median_val = self.df_images['num_detection_objects'].median()

		ax8.axvline(
			mean_val,
			color='red',
			linestyle='--',
			linewidth=2,
			label=f'Mean: {mean_val:.1f}'
		)
		ax8.axvline(
			median_val,
			color='green',
			linestyle='--',
			linewidth=2,
			label=f'Median: {median_val:.1f}'
		)

		ax8.set_xlabel('Number of Objects per Image')
		ax8.set_ylabel('Number of Images')
		ax8.set_title('Distribution of Object Counts per Image')
		ax8.legend()
		ax8.grid(True, alpha=0.3)

	def _create_environmental_diversity_plot(self, fig, gs) -> None:
		"""Create environmental diversity pie chart."""
		ax9 = fig.add_subplot(gs[3, 2])
		weather_dist = self.stats['diversity_stats']['weather_diversity']
		colors = plt.cm.Set3(np.linspace(0, 1, len(weather_dist)))

		wedges, texts, autotexts = ax9.pie(
			weather_dist.values(),
			labels=weather_dist.keys(),
			autopct='%1.1f%%',
			colors=colors,
			startangle=90
		)
		ax9.set_title('Weather Condition Diversity')

		# Make percentage text more readable
		for autotext in autotexts:
			autotext.set_color('white')
			autotext.set_fontsize(9)
			autotext.set_weight('bold')

	def save_interesting_samples(self, samples: Dict, output_dir: Path,
								 images_dir: Path = None) -> None:
		"""Save lists of interesting sample images and optionally copy actual images.

		Args:
			samples: Dictionary containing interesting samples.
			output_dir: Output directory path.
			images_dir: Directory containing actual images (optional).
		"""
		samples_dir = output_dir / 'interesting_samples'
		samples_dir.mkdir(exist_ok=True)

		# Save sample lists as JSON
		for category, sample_list in samples.items():
			if not sample_list:
				continue

			# Save list to JSON
			json_path = samples_dir / f'{category}_samples.json'
			with open(json_path, 'w', encoding='utf-8') as f:
				json.dump(sample_list, f, indent=2)

			# If images directory provided, copy actual images
			if images_dir and images_dir.exists():
				self._copy_sample_images(
					sample_list, samples_dir, category, images_dir
				)

	def _copy_sample_images(self, sample_list: List, samples_dir: Path,
						   category: str, images_dir: Path) -> None:
		"""Copy sample images to output directory.

		Args:
			sample_list: List of samples.
			samples_dir: Samples directory path.
			category: Category name.
			images_dir: Source images directory.
		"""
		category_img_dir = samples_dir / category
		category_img_dir.mkdir(exist_ok=True)

		# Extract image names from samples
		image_names = []
		if isinstance(sample_list, list):
			for item in sample_list:
				if isinstance(item, dict) and 'image_name' in item:
					image_names.append(item['image_name'])
		elif isinstance(sample_list, dict):
			for key, value in sample_list.items():
				if isinstance(value, list):
					for item in value:
						if isinstance(item, dict) and 'image_name' in item:
							image_names.append(item['image_name'])

		# Copy images (limit to 10 per category)
		for img_name in image_names[:10]:
			src_path = images_dir / img_name
			if src_path.exists():
				dst_path = category_img_dir / img_name
				shutil.copy2(src_path, dst_path)
				print(f"  Copied: {img_name} to {category}")

	def generate_detailed_report(self, output_dir: Path) -> Path:
		"""Generate a detailed analysis report.

		Args:
			output_dir: Directory to save the report.

		Returns:
			Path to the generated report file.
		"""
		report_path = output_dir / 'detailed_analysis_report.txt'

		with open(report_path, 'w', encoding='utf-8') as f:
			self._write_report_header(f)
			self._write_overall_statistics(f)
			self._write_class_statistics(f)
			self._write_environmental_diversity(f)
			self._write_cross_scenario_analysis(f)
			self._write_object_distribution_analysis(f)

		print(f"Detailed report saved to: {report_path}")
		return report_path

	def _write_report_header(self, f) -> None:
		"""Write report header."""
		f.write("=" * 100 + "\n")
		f.write("BDD100K OBJECT DETECTION DATASET - DETAILED ANALYSIS REPORT\n")
		f.write("=" * 100 + "\n\n")

	def _write_overall_statistics(self, f) -> None:
		"""Write overall statistics section."""
		f.write("1. OVERALL STATISTICS\n")
		f.write("-" * 50 + "\n")
		f.write(f"Total Images: {self.stats['total_images']:,}\n")
		f.write(f"Total Detection Objects: {self.stats['total_annotations']:,}\n")
		f.write(
			f"Average Objects per Image: "
			f"{self.stats['avg_objects_per_image']:.2f} ± "
			f"{self.stats['std_objects_per_image']:.2f}\n\n"
		)

	def _write_class_statistics(self, f) -> None:
		"""Write per-class statistics section."""
		f.write("2. PER-CLASS DETAILED STATISTICS\n")
		f.write("-" * 50 + "\n")

		for cls in self.DETECTION_CLASSES:
			if cls not in self.stats['class_stats']:
				continue

			cs = self.stats['class_stats'][cls]
			f.write(f"\n{cls.upper()}:\n")
			f.write(
				f"  Total Instances: {cs['count']:,} "
				f"({cs['percentage_of_total']:.2f}% of all objects)\n"
			)
			f.write(f"  Images with {cls}: {cs['images_with_class']:,}\n")
			f.write(f"  Average per Image: {cs['avg_per_image']:.3f}\n")
			f.write(
				f"  Average Size: {cs['avg_width']:.1f} × "
				f"{cs['avg_height']:.1f} pixels\n"
			)
			f.write(
				f"  Average Area: {cs['avg_area']:.1f} px² "
				f"(median: {cs['median_area']:.1f})\n"
			)
			f.write(
				f"  Occlusion: {cs['occluded_count']:,} instances "
				f"({cs['occluded_ratio']:.1f}%)\n"
			)
			f.write(
				f"  Truncation: {cs['truncated_count']:,} instances "
				f"({cs['truncated_ratio']:.1f}%)\n"
			)
			f.write(
				f"  Both Occluded & Truncated: "
				f"{cs['both_occluded_truncated']:,} instances\n"
			)

			if cs.get('size_distribution'):
				f.write("  Size Distribution: ")
				for size, count in sorted(cs['size_distribution'].items()):
					f.write(f"{size}:{count} ")
				f.write("\n")

	def _write_environmental_diversity(self, f) -> None:
		"""Write environmental diversity section."""
		f.write("\n3. ENVIRONMENTAL DIVERSITY\n")
		f.write("-" * 50 + "\n")

		f.write("\nWeather Conditions:\n")
		for weather, pct in sorted(
			self.stats['diversity_stats']['weather_diversity'].items(),
			key=lambda x: x[1],
			reverse=True
		):
			f.write(f"  {weather:15s}: {pct:6.2f}%\n")

		f.write("\nScene Types:\n")
		for scene, pct in sorted(
			self.stats['diversity_stats']['scene_diversity'].items(),
			key=lambda x: x[1],
			reverse=True
		):
			f.write(f"  {scene:15s}: {pct:6.2f}%\n")

		f.write("\nTime of Day:\n")
		for time, pct in sorted(
			self.stats['diversity_stats']['timeofday_diversity'].items(),
			key=lambda x: x[1],
			reverse=True
		):
			f.write(f"  {time:15s}: {pct:6.2f}%\n")

	def _write_cross_scenario_analysis(self, f) -> None:
		"""Write cross-scenario analysis section."""
		f.write("\n4. CROSS-SCENARIO ANALYSIS\n")
		f.write("-" * 50 + "\n")

		f.write("\nChallenging Weather + Time Combinations:\n")
		challenging = self.stats['cross_scenario_analysis']['challenging_scenarios']
		for scenario, count in challenging.items():
			f.write(
				f"  {scenario.replace('_', ' ').title():30s}: "
				f"{count:6,} images\n"
			)

		f.write("\nCity vs Non-City:\n")
		city_stats = self.stats['city_analysis']
		f.write(
			f"  City Streets: {city_stats['city_images']:,} images "
			f"(avg {city_stats['avg_objects_city']:.1f} objects/image)\n"
		)
		f.write(
			f"  Non-City: {city_stats['non_city_images']:,} images "
			f"(avg {city_stats['avg_objects_non_city']:.1f} objects/image)\n"
		)

		f.write("\nDay vs Night:\n")
		day_night = self.stats['day_night_analysis']
		f.write(
			f"  Daytime: {day_night['day_images']:,} images "
			f"(avg {day_night['avg_objects_day']:.1f} objects/image)\n"
		)
		f.write(
			f"  Nighttime: {day_night['night_images']:,} images "
			f"(avg {day_night['avg_objects_night']:.1f} objects/image)\n"
		)
		f.write(f"  Dawn/Dusk: {day_night['dawn_dusk_images']:,} images\n")

	def _write_object_distribution_analysis(self, f) -> None:
		"""Write object distribution analysis section."""
		f.write("\n5. OBJECT DISTRIBUTION IN CHALLENGING CONDITIONS\n")
		f.write("-" * 50 + "\n")

		obj_challenging = self.stats['cross_scenario_analysis'][
			'object_distribution_challenging'
		]

		f.write("Challenging Conditions (night/bad weather):\n")
		f.write(f"  Images: {obj_challenging['total_challenging_images']:,}\n")
		f.write(
			f"  Avg Objects: {obj_challenging['avg_objects_challenging']:.2f}\n"
		)
		f.write("Normal Conditions:\n")
		f.write(f"  Images: {obj_challenging['total_normal_images']:,}\n")
		f.write(f"  Avg Objects: {obj_challenging['avg_objects_normal']:.2f}\n")


def detect_dataset_split(json_path: str, data: list = None) -> str:
	"""Detect which split (train/val/test) the dataset represents.

	Args:
		json_path: Path to the JSON file.
		data: Optional data list to analyze.

	Returns:
		String indicating the detected split type.
	"""
	# Try to detect from filename
	json_path_str = str(json_path).lower()
	if 'train' in json_path_str:
		return 'train'
	elif 'val' in json_path_str:
		return 'val'
	elif 'test' in json_path_str:
		return 'test'

	# Try to detect from data size if data is provided
	if data:
		num_images = len(data)
		if 65000 <= num_images <= 75000:
			return 'train'
		elif 8000 <= num_images <= 12000:
			return 'val'
		elif 18000 <= num_images <= 22000:
			return 'test'

	# Default to unknown
	return 'unknown'


def generate_markdown_report(analyzer: BDD100KAnalyzer, stats: Dict,
						   output_dir: Path, split_name: str) -> Path:
	"""Generate comprehensive markdown report with embedded visualizations.

	Args:
		analyzer: BDD100KAnalyzer instance.
		stats: Statistics dictionary.
		output_dir: Output directory path.
		split_name: Dataset split name.

	Returns:
		Path to the generated markdown report.
	"""
	report_path = output_dir / 'analysis_report.md'

	with open(report_path, 'w', encoding='utf-8') as f:
		# Header
		f.write("# BDD100K Object Detection Dataset Analysis Report\n")
		f.write(f"## Dataset Split: {split_name.upper()}\n")
		f.write(
			f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
		)
		f.write("---\n\n")

		# Table of Contents
		_write_table_of_contents(f)

		# Executive Summary
		_write_executive_summary(f, split_name, stats)

		# Dataset Overview
		_write_dataset_overview(f, stats)

		# Class Distribution Analysis
		_write_class_distribution_analysis(f, analyzer, stats)

		# Environmental Conditions
		_write_environmental_conditions(f, stats)

		# Cross-Scenario Analysis
		_write_cross_scenario_analysis_md(f, stats)

		# Visualizations
		_write_visualizations_section(f)

		# Key Findings
		_write_key_findings(f, stats)

		# Recommendations
		_write_recommendations(f)

		# Footer
		_write_report_footer(f, split_name)

	print(f"Markdown report saved to: {report_path}")
	return report_path


def _write_table_of_contents(f) -> None:
	"""Write table of contents to markdown file."""
	f.write("## Table of Contents\n")
	f.write("1. [Executive Summary](#executive-summary)\n")
	f.write("2. [Dataset Overview](#dataset-overview)\n")
	f.write("3. [Class Distribution Analysis](#class-distribution-analysis)\n")
	f.write("4. [Environmental Conditions](#environmental-conditions)\n")
	f.write("5. [Cross-Scenario Analysis](#cross-scenario-analysis)\n")
	f.write("6. [Anomalies and Edge Cases](#anomalies-and-edge-cases)\n")
	f.write("7. [Visualizations](#visualizations)\n")
	f.write("8. [Key Findings](#key-findings)\n")
	f.write("9. [Recommendations](#recommendations)\n\n")


def _write_executive_summary(f, split_name: str, stats: Dict) -> None:
	"""Write executive summary section."""
	f.write("## Executive Summary\n\n")
	f.write(
		f"This report analyzes the **{split_name}** split of the BDD100K "
		f"object detection dataset, containing **{stats['total_images']:,}** "
		f"images with **{stats['total_annotations']:,}** object annotations "
		f"across 10 categories.\n\n"
	)


def _write_dataset_overview(f, stats: Dict) -> None:
	"""Write dataset overview section."""
	f.write("## Dataset Overview\n\n")
	f.write("### Basic Statistics\n")
	f.write(f"- **Total Images**: {stats['total_images']:,}\n")
	f.write(f"- **Total Annotations**: {stats['total_annotations']:,}\n")
	f.write(
		f"- **Average Objects per Image**: "
		f"{stats['avg_objects_per_image']:.2f} ± "
		f"{stats['std_objects_per_image']:.2f}\n"
	)


def _write_class_distribution_analysis(f, analyzer: BDD100KAnalyzer,
									 stats: Dict) -> None:
	"""Write class distribution analysis section."""
	f.write("## Class Distribution Analysis\n\n")
	f.write("### Object Classes Summary\n\n")
	f.write(
		"| Class | Count | % of Total | Images with Class | "
		"Avg per Image | Occluded % | Truncated % |\n"
	)
	f.write(
		"|-------|-------|------------|-------------------|"
		"---------------|------------|-------------|\n"
	)

	# Sort classes by count
	class_stats_sorted = sorted(
		[
			(cls, stats['class_stats'][cls])
			for cls in analyzer.DETECTION_CLASSES
			if cls in stats['class_stats']
		],
		key=lambda x: x[1]['count'],
		reverse=True
	)

	for cls, cs in class_stats_sorted:
		f.write(
			f"| {cls} | {cs['count']:,} | {cs['percentage_of_total']:.1f}% | "
			f"{cs['images_with_class']:,} | {cs['avg_per_image']:.2f} | "
			f"{cs['occluded_ratio']:.1f}% | {cs['truncated_ratio']:.1f}% |\n"
		)

	f.write("\n### Class Imbalance Analysis\n\n")
	if class_stats_sorted:
		max_class = class_stats_sorted[0]
		min_class = class_stats_sorted[-1]
		imbalance_ratio = (
			max_class[1]['count'] / min_class[1]['count']
			if min_class[1]['count'] > 0 else float('inf')
		)
		f.write(
			f"**Severe class imbalance detected**: {max_class[0]} "
			f"({max_class[1]['count']:,}) vs {min_class[0]} "
			f"({min_class[1]['count']:,}) = **{imbalance_ratio:.0f}:1 ratio**\n\n"
		)


def _write_environmental_conditions(f, stats: Dict) -> None:
	"""Write environmental conditions section."""
	f.write("## Environmental Conditions\n\n")

	f.write("### Weather Distribution\n\n")
	f.write("| Weather | Images | Percentage |\n")
	f.write("|---------|--------|------------|\n")
	for weather, pct in sorted(
		stats['diversity_stats']['weather_diversity'].items(),
		key=lambda x: x[1],
		reverse=True
	):
		count = int(stats['total_images'] * pct / 100)
		f.write(f"| {weather} | {count:,} | {pct:.1f}% |\n")

	f.write("\n### Scene Distribution\n\n")
	f.write("| Scene | Images | Percentage |\n")
	f.write("|-------|--------|------------|\n")
	for scene, pct in sorted(
		stats['diversity_stats']['scene_diversity'].items(),
		key=lambda x: x[1],
		reverse=True
	):
		count = int(stats['total_images'] * pct / 100)
		f.write(f"| {scene} | {count:,} | {pct:.1f}% |\n")

	f.write("\n### Time of Day Distribution\n\n")
	f.write("| Time | Images | Percentage |\n")
	f.write("|------|--------|------------|\n")
	for time, pct in sorted(
		stats['diversity_stats']['timeofday_diversity'].items(),
		key=lambda x: x[1],
		reverse=True
	):
		count = int(stats['total_images'] * pct / 100)
		f.write(f"| {time} | {count:,} | {pct:.1f}% |\n")


def _write_cross_scenario_analysis_md(f, stats: Dict) -> None:
	"""Write cross-scenario analysis section."""
	f.write("\n## Cross-Scenario Analysis\n\n")

	f.write("### Challenging Weather + Time Combinations\n\n")
	challenging = stats['cross_scenario_analysis']['challenging_scenarios']
	f.write("| Scenario | Image Count | % of Dataset |\n")
	f.write("|----------|-------------|---------------|\n")
	for scenario, count in sorted(challenging.items(), key=lambda x: x[1], reverse=True):
		pct = (count / stats['total_images']) * 100
		scenario_name = scenario.replace('_', ' ').title()
		f.write(f"| {scenario_name} | {count:,} | {pct:.2f}% |\n")

	f.write("\n### City vs Non-City Analysis\n\n")
	city_stats = stats['city_analysis']
	f.write(
		f"- **City Streets**: {city_stats['city_images']:,} images "
		f"(avg {city_stats['avg_objects_city']:.1f} objects/image)\n"
	)
	f.write(
		f"- **Non-City**: {city_stats['non_city_images']:,} images "
		f"(avg {city_stats['avg_objects_non_city']:.1f} objects/image)\n"
	)
	f.write(
		f"- **Difference**: City scenes contain "
		f"{abs(city_stats['avg_objects_city'] - city_stats['avg_objects_non_city']):.1f} "
		f"more objects per image on average\n\n"
	)

	f.write("### Day vs Night Analysis\n\n")
	day_night = stats['day_night_analysis']
	f.write(
		f"- **Daytime**: {day_night['day_images']:,} images "
		f"(avg {day_night['avg_objects_day']:.1f} objects/image)\n"
	)
	f.write(
		f"- **Nighttime**: {day_night['night_images']:,} images "
		f"(avg {day_night['avg_objects_night']:.1f} objects/image)\n"
	)
	f.write(f"- **Dawn/Dusk**: {day_night['dawn_dusk_images']:,} images\n\n")


def _write_visualizations_section(f) -> None:
	"""Write visualizations section."""
	f.write("## Visualizations\n\n")
	f.write("### Main Analysis Dashboard\n")
	f.write("![Analysis Dashboard](analysis_visualization.png)\n\n")
	f.write(
		"*Comprehensive visualization showing class distribution, "
		"environmental conditions, and cross-scenario analysis*\n\n"
	)


def _write_key_findings(f, stats: Dict) -> None:
	"""Write key findings section."""
	f.write("## Key Findings\n\n")
	f.write("### Critical Issues Identified\n\n")

	# Check for severe imbalance
	class_stats_sorted = sorted(
		[(cls, cs) for cls, cs in stats['class_stats'].items()],
		key=lambda x: x[1]['count'],
		reverse=True
	)

	if class_stats_sorted:
		f.write("1. **Extreme Class Imbalance**\n")
		f.write(
			f"   - Most common class: {class_stats_sorted[0][0]} "
			f"({class_stats_sorted[0][1]['count']:,} instances)\n"
		)
		f.write(
			f"   - Rarest class: {class_stats_sorted[-1][0]} "
			f"({class_stats_sorted[-1][1]['count']:,} instances)\n"
		)
		f.write(
			"   - Imbalance will significantly impact model performance "
			"on rare classes\n\n"
		)

	# Check for high occlusion
	avg_occlusion = np.mean([cs['occluded_ratio'] for _, cs in class_stats_sorted])
	if avg_occlusion > 30:
		f.write("2. **High Occlusion Rates**\n")
		f.write(f"   - Overall average: {avg_occlusion:.1f}% of objects occluded\n")
		# Find worst cases
		worst_occlusion = sorted(
			class_stats_sorted,
			key=lambda x: x[1]['occluded_ratio'],
			reverse=True
		)[:3]
		for cls, cs in worst_occlusion:
			f.write(f"   - {cls}: {cs['occluded_ratio']:.1f}% occluded\n")
		f.write("\n")

	# Check for underrepresented conditions
	f.write("3. **Underrepresented Conditions**\n")
	if 'foggy' in stats['diversity_stats']['weather_diversity']:
		foggy_pct = stats['diversity_stats']['weather_diversity']['foggy']
		if foggy_pct < 1:
			f.write(f"   - Foggy conditions: Only {foggy_pct:.2f}% of dataset\n")

	challenging = stats['cross_scenario_analysis']['challenging_scenarios']
	total_challenging = sum(challenging.values())
	challenging_pct = (total_challenging / stats['total_images']) * 100
	f.write(f"   - Total challenging conditions: {challenging_pct:.1f}% of dataset\n")
	f.write(
		f"   - Night + bad weather: "
		f"{challenging['challenging_weather_night']:,} images\n\n"
	)


def _write_recommendations(f) -> None:
	"""Write recommendations section."""
	f.write("## Recommendations\n\n")
	f.write("### For Model Training\n\n")
	f.write("1. **Address Class Imbalance**\n")
	f.write("   - Implement weighted loss functions or focal loss\n")
	f.write("   - Use oversampling for rare classes\n")
	f.write("   - Consider class-balanced sampling strategies\n\n")

	f.write("2. **Handle High Occlusion**\n")
	f.write("   - Use data augmentation with artificial occlusion\n")
	f.write("   - Implement robust feature extraction methods\n")
	f.write("   - Consider part-based detection approaches\n\n")

	f.write("3. **Improve Robustness to Challenging Conditions**\n")
	f.write("   - Augment training with synthetic weather effects\n")
	f.write("   - Use domain adaptation techniques\n")
	f.write("   - Develop specialized models for night/adverse weather\n\n")

	f.write("### For Data Collection\n\n")
	f.write("1. Prioritize collection of:\n")
	f.write("   - Foggy weather conditions\n")
	f.write("   - Rare object classes (trains, motorcycles)\n")
	f.write("   - Night + adverse weather combinations\n")
	f.write("   - Dawn/dusk transition periods\n\n")


def _write_report_footer(f, split_name: str) -> None:
	"""Write report footer."""
	f.write("---\n\n")
	f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
	f.write(f"*Dataset: BDD100K {split_name} split*\n")
	f.write("*Total processing time: See console output*\n")


def main() -> Tuple[BDD100KAnalyzer, Dict]:
	"""Main execution function.

	Returns:
		Tuple containing analyzer instance and computed statistics.
	"""
	parser = argparse.ArgumentParser(
		description='Analyze BDD100K Object Detection Dataset'
	)
	parser.add_argument(
		'--json_path',
		type=str,
		required=True,
		help='Path to BDD100K annotations JSON file'
	)
	parser.add_argument(
		'--images_dir',
		type=str,
		default=None,
		help='Path to BDD100K images directory (optional, for copying sample images)'
	)
	parser.add_argument(
		'--output_dir',
		type=str,
		default='./bdd100k_analysis_results',
		help='Directory to save analysis results'
	)
	parser.add_argument(
		'--sample_size',
		type=int,
		default=None,
		help='Sample size for testing (default: use all data)'
	)
	parser.add_argument(
		'--generate_pdf',
		action='store_true',
		help='Generate PDF report from markdown (requires pandoc)'
	)

	args = parser.parse_args()

	# Detect dataset split
	print("\nDetecting dataset split...")
	with open(args.json_path, 'r', encoding='utf-8') as f:
		data_peek = json.load(f)
		if args.sample_size:
			data_peek = data_peek[:args.sample_size]

	split_name = detect_dataset_split(args.json_path, data_peek)
	print(f"Detected split: {split_name}")

	# Create output directory with split suffix
	base_output_dir = Path(args.output_dir)
	if split_name != 'unknown':
		output_dir = Path(f"{base_output_dir}_{split_name}")
	else:
		output_dir = base_output_dir

	output_dir.mkdir(parents=True, exist_ok=True)

	print("=" * 100)
	print("BDD100K OBJECT DETECTION ANALYSIS")
	print("=" * 100)
	print(f"Start time: {datetime.now()}")
	print(f"JSON path: {args.json_path}")
	print(f"Dataset split: {split_name.upper()}")
	print(f"Images directory: {args.images_dir if args.images_dir else 'Not provided'}")
	print(f"Output directory: {output_dir}")
	if args.sample_size:
		print(f"Sample size: {args.sample_size}")
	print("-" * 100)

	# Initialize analyzer
	print("\nInitializing analyzer...")
	analyzer = BDD100KAnalyzer(args.json_path)

	# Load annotations
	print("Loading annotations...")
	start_time = time.time()
	analyzer.load_annotations(sample_size=args.sample_size)
	load_time = time.time() - start_time
	print(f"Loading completed in {load_time:.2f} seconds")

	# Parse to DataFrames
	print("\nParsing annotations to DataFrames...")
	start_time = time.time()
	analyzer.parse_to_dataframes()
	parse_time = time.time() - start_time
	print(f"Parsing completed in {parse_time:.2f} seconds")

	# Compute comprehensive statistics
	print("\nComputing comprehensive statistics...")
	analyzer.stats = analyzer.compute_comprehensive_statistics()

	# Print summary
	_print_dataset_summary(analyzer, split_name)

	# Find interesting samples
	print("\nFinding interesting samples...")
	interesting_samples = analyzer.find_interesting_samples()

	# Save interesting samples (with optional image copying)
	print("Saving interesting samples...")
	if args.images_dir:
		images_dir = Path(args.images_dir)
		analyzer.save_interesting_samples(interesting_samples, output_dir, images_dir)
	else:
		analyzer.save_interesting_samples(interesting_samples, output_dir)

	# Create visualizations
	print("\nCreating visualizations...")
	viz_path = analyzer.create_visualizations(output_dir)
	print(f"Visualizations saved to: {viz_path}")

	# Generate markdown report
	print("\nGenerating markdown report...")
	markdown_path = generate_markdown_report(
		analyzer, analyzer.stats, output_dir, split_name
	)

	# Generate PDF if requested and pandoc is available
	if args.generate_pdf:
		_generate_pdf_report(output_dir, markdown_path)

	# Generate detailed text report
	print("\nGenerating detailed text report...")
	report_path = analyzer.generate_detailed_report(output_dir)

	# Save raw statistics to JSON
	_save_statistics_json(analyzer.stats, output_dir)

	# Export DataFrames to CSV for further analysis
	_export_dataframes_and_crosstabs(analyzer, output_dir)

	# Print completion summary
	_print_completion_summary(output_dir, viz_path, markdown_path, args.generate_pdf)

	# Print key insights
	_print_key_insights(analyzer, split_name)

	return analyzer, analyzer.stats


def _print_dataset_summary(analyzer: BDD100KAnalyzer, split_name: str) -> None:
	"""Print dataset summary to console."""
	stats = analyzer.stats

	print("\n" + "=" * 100)
	print(f"DATASET SUMMARY - {split_name.upper()} SPLIT")
	print("=" * 100)

	print(f"Total Images: {stats['total_images']:,}")
	print(f"Total Detection Objects: {stats['total_annotations']:,}")
	print(f"Average Objects per Image: {stats['avg_objects_per_image']:.2f}")

	print("\nClass Distribution (Top 5):")
	print("-" * 50)
	class_counts = [
		(cls, stats['class_stats'][cls]['count'])
		for cls in analyzer.DETECTION_CLASSES
		if cls in stats['class_stats']
	]
	class_counts.sort(key=lambda x: x[1], reverse=True)

	for cls, count in class_counts[:5]:
		pct = stats['class_stats'][cls]['percentage_of_total']
		occluded = stats['class_stats'][cls]['occluded_ratio']
		print(
			f"  {cls:15s}: {count:7,} instances ({pct:5.2f}%) - "
			f"{occluded:.1f}% occluded"
		)

	print("\nChallenging Scenarios:")
	print("-" * 50)
	challenging = stats['cross_scenario_analysis']['challenging_scenarios']
	for scenario, count in challenging.items():
		print(f"  {scenario.replace('_', ' ').title():30s}: {count:6,} images")


def _generate_pdf_report(output_dir: Path, markdown_path: Path) -> None:
	"""Generate PDF report from markdown."""
	try:
		pdf_path = output_dir / 'analysis_report.pdf'

		print("\nGenerating PDF report...")
		# Using pandoc to convert markdown to PDF
		cmd = [
			'pandoc',
			str(markdown_path),
			'-o', str(pdf_path),
			'--pdf-engine=xelatex',
			'-V', 'geometry:margin=1in',
			'-V', 'colorlinks=true',
			'-V', 'linkcolor=blue',
			'-V', 'urlcolor=blue',
			'--highlight-style=kate'
		]

		result = subprocess.run(cmd, capture_output=True, text=True, check=False)

		if result.returncode == 0:
			print(f"PDF report saved to: {pdf_path}")
		else:
			print("PDF generation failed. Ensure pandoc and xelatex are installed.")
			print(f"Error: {result.stderr}")
	except FileNotFoundError:
		print("PDF generation skipped. Install pandoc to generate PDF reports:")
		print("  Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex")
		print("  MacOS: brew install pandoc mactex")
		print("  Windows: Download from https://pandoc.org/installing.html")


def _save_statistics_json(stats: Dict, output_dir: Path) -> None:
	"""Save statistics to JSON file."""
	stats_file = output_dir / 'comprehensive_statistics.json'

	def convert_types(obj):
		"""Convert numpy types to Python types for JSON serialization."""
		if isinstance(obj, (np.integer, np.int64)):
			return int(obj)
		elif isinstance(obj, (np.floating, np.float64)):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, dict):
			# Convert any tuple keys to strings
			converted_dict = {}
			for k, v in obj.items():
				if isinstance(k, tuple):
					# Convert tuple key to string
					key = '_'.join(str(x) for x in k)
				else:
					key = k
				converted_dict[key] = convert_types(v)
			return converted_dict
		elif isinstance(obj, list):
			return [convert_types(item) for item in obj]
		elif isinstance(obj, tuple):
			return list(obj)  # Convert tuples to lists
		return obj

	with open(stats_file, 'w', encoding='utf-8') as f:
		json.dump(convert_types(stats), f, indent=2)
	print(f"Statistics JSON saved to: {stats_file}")


def _export_dataframes_and_crosstabs(analyzer: BDD100KAnalyzer,
									output_dir: Path) -> None:
	"""Export DataFrames and cross-tabulations to CSV."""
	print("\nExporting DataFrames to CSV...")
	analyzer.df_annotations.to_csv(output_dir / 'annotations.csv', index=False)
	analyzer.df_images.to_csv(output_dir / 'images.csv', index=False)

	# Create cross-tabulation CSV files
	cross_tabs_dir = output_dir / 'cross_tabulations'
	cross_tabs_dir.mkdir(exist_ok=True)

	# Weather × Time cross-tab
	weather_time = pd.crosstab(
		analyzer.df_images['weather'],
		analyzer.df_images['timeofday']
	)
	weather_time.to_csv(cross_tabs_dir / 'weather_x_timeofday.csv')

	# Scene × Time cross-tab
	scene_time = pd.crosstab(
		analyzer.df_images['scene'],
		analyzer.df_images['timeofday']
	)
	scene_time.to_csv(cross_tabs_dir / 'scene_x_timeofday.csv')

	# Class × Weather cross-tab
	class_weather = pd.crosstab(
		analyzer.df_annotations['category'],
		analyzer.df_annotations['weather']
	)
	class_weather.to_csv(cross_tabs_dir / 'class_x_weather.csv')

	print(f"Cross-tabulation files saved to: {cross_tabs_dir}")


def _print_completion_summary(output_dir: Path, viz_path: Path,
							markdown_path: Path, generate_pdf: bool) -> None:
	"""Print completion summary."""
	print("\n" + "=" * 100)
	print("ANALYSIS COMPLETE")
	print("=" * 100)
	print(f"End time: {datetime.now()}")
	print(f"All results saved to: {output_dir}")
	print("\nKey outputs:")
	print(f"  📊 Visualization: {viz_path.name}")
	print(f"  📝 Markdown report: {markdown_path.name}")
	if generate_pdf and (output_dir / 'analysis_report.pdf').exists():
		print("  📄 PDF report: analysis_report.pdf")
	print(f"  📈 Statistics: comprehensive_statistics.json")
	print(f"  📁 Interesting samples: interesting_samples/")


def _print_key_insights(analyzer: BDD100KAnalyzer, split_name: str) -> None:
	"""Print key insights."""
	stats = analyzer.stats

	print("\nKEY INSIGHTS:")
	print("-" * 50)

	# Class imbalance
	class_counts = [
		(cls, stats['class_stats'][cls]['count'])
		for cls in analyzer.DETECTION_CLASSES
		if cls in stats['class_stats']
	]
	class_counts.sort(key=lambda x: x[1], reverse=True)

	max_class = max(class_counts, key=lambda x: x[1])
	min_class = min(class_counts, key=lambda x: x[1])
	print(
		f"• Severe class imbalance: {max_class[0]} ({max_class[1]:,}) vs "
		f"{min_class[0]} ({min_class[1]:,})"
	)

	# Challenging conditions
	challenging = stats['cross_scenario_analysis']['challenging_scenarios']
	total_challenging = sum(challenging.values())
	pct_challenging = (total_challenging / stats['total_images']) * 100
	print(
		f"• Challenging conditions: {total_challenging:,} images "
		f"({pct_challenging:.1f}% of dataset)"
	)

	# Occlusion prevalence
	total_occluded = sum(
		stats['class_stats'][cls]['occluded_count']
		for cls in analyzer.DETECTION_CLASSES
		if cls in stats['class_stats']
	)
	pct_occluded = (total_occluded / stats['total_annotations']) * 100
	print(f"• Overall occlusion rate: {pct_occluded:.1f}% of all objects")

	# City vs Non-city difference
	city_diff = abs(
		stats['city_analysis']['avg_objects_city'] -
		stats['city_analysis']['avg_objects_non_city']
	)
	print(f"• City scenes have {city_diff:.1f} more objects per image on average")

	print(f"\n• Dataset split '{split_name}' contains {stats['total_images']:,} images")


if __name__ == "__main__":
	main()