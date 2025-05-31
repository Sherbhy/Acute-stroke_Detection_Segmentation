# Acute Stroke Detection & Segmentation (ADS) - Enhanced Edition

## Overview

This is an enhanced version of the Acute Stroke Detection & Segmentation tool, featuring significant performance optimizations, batch processing capabilities, and comprehensive validation tools. The original deep learning networks were trained and tested on 2,348 clinical diffusion weighted MRI images, with external validation on 280 additional images.

**Enhanced by:** Surbhi Agarwal  
**Original Work:** Liu et al. (2021) - Deep learning-based detection and segmentation of diffusion abnormalities in acute ischemic stroke

## Key Enhancements

### Performance Optimizations
- **GPU Auto-Detection**: Automatic GPU configuration with fallback to CPU
- **Model Caching**: Intelligent model loading and memory management
- **Memory Optimization**: Reduced memory footprint for clinical deployment
- **Progress Tracking**: Real-time performance monitoring and progress bars

### New Tools & Scripts
- **Batch Processing**: Process multiple subjects automatically (`batch_process_ads.py`)
- **Results Validation**: Comprehensive quality assessment (`validate_results.py`)
- **Setup Automation**: Environment validation and setup (`setup_ads.py`)
- **Interactive Guide**: Usage examples and troubleshooting (`usage_guide.py`)
- **Enhanced CLI**: Improved command-line interface with extensive options

### Clinical Workflow Improvements
- **Automated Results Organization**: Timestamped results folders
- **Quality Control**: Automated lesion metrics and quality flags
- **Statistical Analysis**: Batch validation with summary reports and plots
- **Error Handling**: Robust error handling with detailed logging

## Quick Start

### 1. Environment Setup
```bash
# Validate and setup environment
python setup_ads.py --install-packages --check-only

# For interactive setup guidance
python usage_guide.py --interactive
```

### 2. Single Subject Processing
```bash
# Basic processing
python codes/ads_run_optimizations.py -input "data/examples/Subject01/"

# Advanced processing with progress tracking
python codes/ads_run_optimizations.py \
    -input "data/examples/Subject01/" \
    -model DAGMNet_CH3 \
    -show_progress \
    --verbose
```

### 3. Batch Processing
```bash
# Sequential batch processing
python batch_process_ads.py /path/to/subjects/ -verbose

# Parallel processing for large datasets
python batch_process_ads.py /path/to/subjects/ \
    -parallel -workers 4 -verbose
```

### 4. Results Validation
```bash
# Validate single result
python validate_results.py /path/to/results/ -verbose

# Batch validation with reports
python validate_results.py /path/to/subjects/ \
    -batch -report -plots
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, but recommended)

### Step 1: Download and Setup
```bash
# Clone repository
git clone https://github.com/Chin-Fu-Liu/Acute-stroke_Detection_Segmentation/

# Or download from Google Drive
# [Download link from original repository]

# Navigate to directory
cd Acute-stroke_Detection_Segmentation/
```

### Step 2: Environment Setup
```bash
# Automated setup (recommended)
python setup_ads.py --install-packages

# Manual setup
pip install -r requirements.txt
```

### Step 3: Download Models
Download pre-trained models from:
- [NITRC Repository](https://www.nitrc.org/projects/ads)
- [Zenodo](https://zenodo.org/record/5579390)

Place `.h5` files in `data/Trained_Nets/` directory.

## Enhanced Features

### Batch Processing
Process multiple subjects with automatic error handling:
```bash
python batch_process_ads.py /path/to/study_data/ \
    -model DAGMNet_CH3 \
    -parallel -workers 2 \
    -verbose
```

### Results Validation
Comprehensive quality assessment with statistical analysis:
```bash
python validate_results.py /path/to/results/ \
    -batch -report -plots \
    -output validation_reports/
```

### Performance Monitoring
Track processing performance and system resources:
```bash
python codes/ads_run_optimizations.py \
    -input "data/examples/Subject01/" \
    -show_progress \
    -clear_models
```

## File Structure

```
${ROOT}
|-- codes/
    |-- ads_run_optimizations.py      # Enhanced main script
    |-- ads_bin_optimizations.py      # Optimized core functions
    |-- ADS_bin.py                     # Original core functions
    |-- ADSRun.py                      # Original main script
|-- data/
    |-- Trained_Nets/                 # Pre-trained models
    |-- examples/                     # Example data
    |-- template/                     # MNI templates and atlases
|-- batch_process_ads.py              # Batch processing
|-- validate_results.py               # Results validation
|-- setup_ads.py                      # Environment setup
|-- usage_guide.py                    # Interactive guide
|-- requirements.txt                  # Dependencies
```

## Input Data Format

Subject folder structure:
```
SubjectID_Folder/
├── SubjectID_DWI.nii.gz      # Required: Diffusion weighted image
├── SubjectID_b0.nii.gz       # Required: B0 reference image
└── SubjectID_ADC.nii.gz      # Optional: ADC map (calculated if missing)
```

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| DAGMNet_CH3 | Best performance (default) | Clinical use, research |
| DAGMNet_CH2 | Faster processing | Large studies |
| UNet_CH3 | Good alternative | Backup option |
| UNet_CH2 | Lightweight | Resource-limited environments |
| FCN_CH3/CH2 | Basic functionality | Comparison studies |

## Output Files

### Standard Outputs
- `SubjectID_ModelName_Lesion_Predict.nii.gz` - Binary lesion mask
- `SubjectID_volume_brain_regions.txt` - Detailed lesion report
- `SubjectID_ModelName_Lesion_Predict_result.png` - Visualization

### Optional Outputs (with `-save_MNI`)
- `SubjectID_DWI_MNI.nii.gz` - DWI in MNI space
- `SubjectID_ADC_MNI.nii.gz` - ADC in MNI space
- `SubjectID_b0_MNI.nii.gz` - B0 in MNI space

## Interactive Usage Guide

For comprehensive examples and troubleshooting:
```bash
python usage_guide.py --interactive
```

Available guide sections:
- Quick Start Guide
- Usage Examples (basic to advanced)
- Troubleshooting
- Best Practices
- Clinical Workflow Examples

## Performance Specifications

### Processing Speed
- **CPU**: ~30 seconds lesion inference, ~2.5 minutes total
- **GPU**: ~6 seconds lesion inference, ~1 minute total
- **Batch**: Automatic optimization based on system resources

### System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, GPU with 6GB+ VRAM
- **Batch Processing**: Additional 4GB RAM per parallel worker

## Validation and Quality Control

The enhanced version includes comprehensive validation:

### Automatic Quality Checks
- File completeness verification
- Lesion volume analysis
- Connected components assessment
- Spatial distribution evaluation
- Intensity pattern validation

### Statistical Reports
- Volume distribution analysis
- Success/failure rates
- Processing time statistics
- Quality flag summaries

## Clinical Research Workflow

Complete workflow for research studies:
```bash
# 1. Validate environment
python setup_ads.py --check-only

# 2. Process all subjects
python batch_process_ads.py /path/to/study_data/ \
    -model DAGMNet_CH3 -verbose

# 3. Validate results
python validate_results.py /path/to/study_data/ \
    -batch -report -plots

# 4. Review validation reports
# Check generated CSV files and plots
```

## Troubleshooting

Common issues and solutions:

**GPU Not Detected**: System automatically falls back to CPU processing
**Memory Errors**: Use `--no_save_MNI` flag or reduce parallel workers
**Missing Models**: Download from NITRC repository
**Import Errors**: Run `python setup_ads.py --install-packages`

For comprehensive troubleshooting: `python usage_guide.py --troubleshooting`

## Performance Benchmarks

Based on enhanced implementation:
- **Single Subject**: 30s-2min depending on hardware
- **Batch (10 subjects)**: 5-20min with parallel processing
- **Large Study (100+ subjects)**: Scales linearly with parallel processing

## References

**Original Method:**
Liu, C.F., Hsu, J., Xu, X., et al. Deep learning-based detection and segmentation of diffusion abnormalities in acute ischemic stroke. *Communications Medicine* 1, 61 (2021). https://doi.org/10.1038/s43856-021-00062-8

**Enhanced Implementation:**
Surbhi Agarwal. Automated Stroke Lesion Detection and Segmentation: Implementation and Evaluation. *Implementation Report*, May 2025.

## Original Repository

For the original implementation and additional resources:
- **GitHub**: https://github.com/Chin-Fu-Liu/Acute-stroke_Detection_Segmentation/
- **NITRC**: https://www.nitrc.org/projects/ads
- **Zenodo**: https://zenodo.org/record/5579390

## License

This enhanced version maintains the original GNU General Public License v3.0. See LICENSE file for details.

## Support

For issues with the enhanced features:
1. Check the interactive guide: `python usage_guide.py --interactive`
2. Validate your environment: `python setup_ads.py --check-only`
3. Review troubleshooting section: `python usage_guide.py --troubleshooting`

For original implementation issues, refer to the original repository documentation.