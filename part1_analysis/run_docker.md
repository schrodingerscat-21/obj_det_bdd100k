

# BDD100K Object Detection Dataset Analysis

A comprehensive analysis tool for the BDD100K object detection dataset with Docker support for easy deployment and reproducible results.

## Prerequisites

- Docker and Docker Compose installed on your system
- BDD100K dataset JSON annotation files
- (Optional) BDD100K image files for sample image extraction

## Quick Start

### 1. Setup Project Structure

```bash
# Clone or create the project directory
mkdir bdd100k-analysis && cd bdd100k-analysis

# Create required directories
mkdir -p data images output

# Copy your BDD100K files
# - Place annotation JSON files in ./data/
# - Place image files in ./images/ (optional)
```

### 2. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Run the analysis
docker-compose up bdd100k-analyzer
```

### 3. View Results

After completion, check the `./output/` directory for:
- `analysis_visualization.png` - Main dashboard visualization
- `analysis_report.md` - Comprehensive markdown report  
- `detailed_analysis_report.txt` - Detailed text report
- `comprehensive_statistics.json` - Raw statistics
- `interesting_samples/` - Sample images and metadata
- `cross_tabulations/` - CSV files with cross-tabulation data


## Usage Options

### Basic Usage

```bash
# Analyze train split
docker-compose run --rm bdd100k-analyzer \
  --json_path /app/data/bdd100k_labels_images_train.json \
  --output_dir /app/output

# Analyze with sample size (for testing)
docker-compose run --rm bdd100k-analyzer \
  --json_path /app/data/bdd100k_labels_images_val.json \
  --output_dir /app/output \
  --sample_size 1000

# Include image copying for samples
docker-compose run --rm bdd100k-analyzer \
  --json_path /app/data/bdd100k_labels_images_val.json \
  --output_dir /app/output \
  --images_dir /app/images
```

### Advanced Usage

```bash
# Generate PDF report (requires pandoc in container)
docker-compose run --rm bdd100k-analyzer \
  --json_path /app/data/bdd100k_labels_images_train.json \
  --output_dir /app/output \
  --generate_pdf

# Development mode with interactive shell
docker-compose run --rm bdd100k-analyzer-dev

# Custom output directory
docker-compose run --rm bdd100k-analyzer \
  --json_path /app/data/bdd100k_labels_images_test.json \
  --output_dir /app/output/test_analysis
```

### Running Without Docker

If you prefer to run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python analysis_new_refactored.py \
  --json_path ./data/bdd100k_labels_images_train.json \
  --output_dir ./output \
  --images_dir ./images
```


## Output Description

### Visualizations
- **Class Distribution**: Log-scale bar chart of object classes
- **Occlusion Analysis**: Per-class occlusion and truncation rates  
- **Weather-Time Heatmap**: Cross-analysis of environmental conditions
- **City vs Non-City**: Comparative analysis of urban vs non-urban scenes
- **Challenging Scenarios**: Distribution of difficult conditions
- **Size Distribution**: Object size categories by class
- **Environmental Diversity**: Weather condition distribution

### Reports
- **Markdown Report**: Comprehensive analysis with tables and insights
- **Text Report**: Detailed statistical breakdown
- **JSON Statistics**: Raw computational results for further analysis

### Data Exports
- **annotations.csv**: Object-level data with environmental context
- **images.csv**: Image-level data with metadata
- **Cross-tabulations**: Weather×Time, Scene×Time, Class×Weather relationships