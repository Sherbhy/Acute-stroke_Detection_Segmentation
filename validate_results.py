#!/usr/bin/env python3
"""
Results Validation Script for Acute Stroke Detection & Segmentation
This script validates and analyzes ADS processing results with comprehensive quality checks,
lesion metrics analysis, and automated report generation for both single subjects and batch studies.
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pandas as pd
from datetime import datetime
import seaborn as sns
from pathlib import Path

class ResultsValidator:
    """Comprehensive validation and analysis of ADS results"""
    
    def __init__(self):
        self.validation_results = []
        self.quality_flags = {}
        
    def validate_single_result(self, results_dir, verbose=False):
        """
        Validate and analyze a single result directory
        
        Args:
            results_dir: Path to the results folder
            verbose: Show detailed output
            
        Returns:
            Dict with validation results
        """
        
        if verbose:
            print(f"Validating: {results_dir}")
        
        # Extract subject ID from folder name
        folder_name = os.path.basename(results_dir.rstrip('/'))
        if '_ADS_Results_' in folder_name:
            subj_id = folder_name.split('_ADS_Results_')[0]
        else:
            subj_id = folder_name
        
        validation_result = {
            'subject_id': subj_id,
            'results_dir': results_dir,
            'validation_time': datetime.now().isoformat(),
            'files_present': {},
            'lesion_metrics': {},
            'quality_flags': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # 1. Check for required files
            self._check_required_files(results_dir, subj_id, validation_result, verbose)
            
            # 2. Analyze lesion mask if present
            lesion_path = self._find_lesion_file(results_dir, subj_id)
            if lesion_path:
                self._analyze_lesion_mask(lesion_path, validation_result, verbose)
            
            # 3. Analyze volume report if present
            report_path = self._find_report_file(results_dir, subj_id)
            if report_path:
                self._analyze_volume_report(report_path, validation_result, verbose)
            
            # 4. Quality checks
            self._perform_quality_checks(validation_result, verbose)
            
            # 5. Overall validation status
            validation_result['status'] = self._determine_validation_status(validation_result)
            
        except Exception as e:
            validation_result['status'] = 'error'
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            if verbose:
                print(f"Validation error: {e}")
        
        return validation_result
    
    def _check_required_files(self, results_dir, subj_id, validation_result, verbose):
        """Check for presence of expected output files"""
        
        # Define expected files
        expected_files = {
            'lesion_mask': [f"{subj_id}_DAGMNet_CH3_Lesion_Predict.nii.gz"],
            'lesion_mask_mni': [f"{subj_id}_DAGMNet_CH3_Lesion_Predict_MNI.nii.gz"],
            'volume_report': [f"{subj_id}_volume_brain_regions.txt"],
            'visualization': [f"{subj_id}_DAGMNet_CH3_Lesion_Predict_result.png"],
            'dwi_mni': [f"{subj_id}_DWI_MNI.nii.gz"],
            'adc_mni': [f"{subj_id}_ADC_MNI.nii.gz"],
            'b0_mni': [f"{subj_id}_b0_MNI.nii.gz"]
        }
        
        files_present = {}
        missing_files = []
        
        for file_type, file_patterns in expected_files.items():
            found = False
            file_size = 0
            
            for pattern in file_patterns:
                file_path = os.path.join(results_dir, pattern)
                if os.path.exists(file_path):
                    found = True
                    file_size = os.path.getsize(file_path)
                    break
            
            files_present[file_type] = {
                'present': found,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024) if found else 0
            }
            
            if not found:
                missing_files.append(file_type)
        
        validation_result['files_present'] = files_present
        
        if missing_files:
            validation_result['warnings'].append(f"Missing files: {', '.join(missing_files)}")
        
        if verbose:
            print(f"   Files check: {len(expected_files) - len(missing_files)}/{len(expected_files)} present")
            if missing_files:
                print(f"      Missing: {', '.join(missing_files)}")
    
    def _find_lesion_file(self, results_dir, subj_id):
        """Find the lesion mask file"""
        patterns = [
            f"{subj_id}_*_Lesion_Predict.nii.gz",
            f"{subj_id}_lesion.nii.gz",
            f"{subj_id}_mask.nii.gz"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(results_dir, pattern))
            if files:
                return files[0]
        return None
    
    def _find_report_file(self, results_dir, subj_id):
        """Find the volume report file"""
        patterns = [
            f"{subj_id}_volume_brain_regions.txt",
            f"{subj_id}_report.txt"
        ]
        
        for pattern in patterns:
            file_path = os.path.join(results_dir, pattern)
            if os.path.exists(file_path):
                return file_path
        return None
    
    def _analyze_lesion_mask(self, lesion_path, validation_result, verbose):
        """Analyze the lesion mask in detail"""
        
        try:
            # Load lesion data
            lesion_img = nib.load(lesion_path)
            lesion_data = lesion_img.get_fdata()
            
            # Basic properties
            image_shape = lesion_data.shape
            voxel_dims = lesion_img.header.get_zooms()[:3]
            voxel_volume = np.prod(voxel_dims)
            
            # Lesion metrics
            lesion_voxels = np.sum(lesion_data > 0.5)
            total_voxels = np.prod(lesion_data.shape)
            lesion_volume_mm3 = lesion_voxels * voxel_volume
            lesion_volume_ml = lesion_volume_mm3 / 1000
            lesion_percentage = (lesion_voxels / total_voxels) * 100
            
            # Connected components analysis
            labeled_lesions, num_components = label(lesion_data > 0.5)
            component_sizes = []
            if num_components > 0:
                for i in range(1, num_components + 1):
                    size = np.sum(labeled_lesions == i)
                    component_sizes.append(size)
                component_sizes.sort(reverse=True)
            
            # Slice distribution
            slices_with_lesion = []
            slice_lesion_volumes = []
            for z in range(lesion_data.shape[2]):
                slice_voxels = np.sum(lesion_data[:, :, z] > 0.5)
                if slice_voxels > 0:
                    slices_with_lesion.append(z)
                    slice_lesion_volumes.append(slice_voxels)
            
            # Intensity analysis
            lesion_intensities = lesion_data[lesion_data > 0.5]
            
            # Store metrics
            validation_result['lesion_metrics'] = {
                'image_shape': image_shape,
                'voxel_dimensions_mm': voxel_dims,
                'voxel_volume_mm3': voxel_volume,
                'lesion_voxels': int(lesion_voxels),
                'total_voxels': int(total_voxels),
                'lesion_volume_mm3': float(lesion_volume_mm3),
                'lesion_volume_ml': float(lesion_volume_ml),
                'lesion_percentage': float(lesion_percentage),
                'num_components': int(num_components),
                'largest_component_voxels': int(component_sizes[0]) if component_sizes else 0,
                'affected_slices': len(slices_with_lesion),
                'total_slices': int(lesion_data.shape[2]),
                'slice_range': [int(min(slices_with_lesion)), int(max(slices_with_lesion))] if slices_with_lesion else [0, 0],
                'mean_intensity': float(np.mean(lesion_intensities)) if len(lesion_intensities) > 0 else 0,
                'std_intensity': float(np.std(lesion_intensities)) if len(lesion_intensities) > 0 else 0,
                'min_intensity': float(np.min(lesion_intensities)) if len(lesion_intensities) > 0 else 0,
                'max_intensity': float(np.max(lesion_intensities)) if len(lesion_intensities) > 0 else 0
            }
            
            if verbose:
                print(f"   Lesion analysis:")
                print(f"      Volume: {lesion_volume_ml:.2f} ml ({lesion_voxels:,} voxels)")
                print(f"      Components: {num_components}")
                print(f"      Affected slices: {len(slices_with_lesion)}/{lesion_data.shape[2]}")
                if slices_with_lesion:
                    print(f"      Slice range: {min(slices_with_lesion)} - {max(slices_with_lesion)}")
        
        except Exception as e:
            validation_result['errors'].append(f"Lesion analysis failed: {str(e)}")
            if verbose:
                print(f"      Lesion analysis error: {e}")
    
    def _analyze_volume_report(self, report_path, validation_result, verbose):
        """Analyze the volume report file"""
        
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            report_data = {}
            
            # Parse key information
            for line in lines:
                if 'intracranial volume' in line.lower():
                    try:
                        icv = int(line.split('\t')[1])
                        report_data['intracranial_volume'] = icv
                    except:
                        pass
                elif 'stroke volume' in line.lower():
                    try:
                        sv = int(line.split('\t')[1])
                        report_data['stroke_volume'] = sv
                    except:
                        pass
            
            # Parse vascular territories
            vascular_territories = {}
            in_vascular_section = False
            for line in lines:
                if 'vascular territory 2' in line.lower():
                    in_vascular_section = True
                    continue
                elif 'area' in line.lower() and 'number of voxel' in line.lower():
                    in_vascular_section = False
                    continue
                
                if in_vascular_section and '\t' in line:
                    try:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            territory = parts[0].strip()
                            count = int(parts[1])
                            vascular_territories[territory] = count
                    except:
                        pass
            
            validation_result['report_data'] = {
                'content_length': len(content),
                'line_count': len(lines),
                'parsed_data': report_data,
                'vascular_territories': vascular_territories
            }
            
            if verbose and report_data:
                print(f"   Report analysis:")
                if 'intracranial_volume' in report_data:
                    print(f"      ICV: {report_data['intracranial_volume']:,} voxels")
                if 'stroke_volume' in report_data:
                    print(f"      Stroke: {report_data['stroke_volume']:,} voxels")
                if vascular_territories:
                    print(f"      Territories: {len(vascular_territories)} analyzed")
        
        except Exception as e:
            validation_result['errors'].append(f"Report analysis failed: {str(e)}")
            if verbose:
                print(f"      Report analysis error: {e}")
    
    def _perform_quality_checks(self, validation_result, verbose):
        """Perform quality checks and set flags"""
        
        quality_flags = {}
        
        # Check if lesion metrics are available
        if 'lesion_metrics' in validation_result:
            metrics = validation_result['lesion_metrics']
            
            # Volume checks
            volume_ml = metrics.get('lesion_volume_ml', 0)
            
            if volume_ml == 0:
                quality_flags['no_lesion_detected'] = True
                validation_result['warnings'].append("No lesion detected")
            elif volume_ml < 0.1:
                quality_flags['very_small_lesion'] = True
                validation_result['warnings'].append(f"Very small lesion: {volume_ml:.3f} ml")
            elif volume_ml > 200:
                quality_flags['very_large_lesion'] = True
                validation_result['warnings'].append(f"Very large lesion: {volume_ml:.1f} ml")
            else:
                quality_flags['reasonable_volume'] = True
            
            # Component checks
            num_components = metrics.get('num_components', 0)
            if num_components > 10:
                quality_flags['many_components'] = True
                validation_result['warnings'].append(f"Many disconnected components: {num_components}")
            elif num_components == 1:
                quality_flags['single_component'] = True
            
            # Spatial distribution checks
            affected_slices = metrics.get('affected_slices', 0)
            total_slices = metrics.get('total_slices', 1)
            
            if affected_slices == 1:
                quality_flags['single_slice_lesion'] = True
                validation_result['warnings'].append("Lesion only in one slice")
            elif affected_slices > total_slices * 0.8:
                quality_flags['spans_many_slices'] = True
                validation_result['warnings'].append("Lesion spans too many slices")
            else:
                quality_flags['reasonable_distribution'] = True
            
            # Intensity checks
            mean_intensity = metrics.get('mean_intensity', 0)
            if mean_intensity > 0.99:
                quality_flags['saturated_intensity'] = True
                validation_result['warnings'].append("Potentially saturated intensities")
        
        # File completeness check
        files_present = validation_result.get('files_present', {})
        required_files = ['lesion_mask', 'volume_report', 'visualization']
        missing_required = [f for f in required_files if not files_present.get(f, {}).get('present', False)]
        
        if not missing_required:
            quality_flags['complete_output'] = True
        else:
            quality_flags['incomplete_output'] = True
        
        validation_result['quality_flags'] = quality_flags
        
        if verbose:
            print(f"   Quality checks:")
            for flag, value in quality_flags.items():
                if value:
                    status = "WARNING" if flag.startswith(('very_', 'many_', 'single_', 'spans_', 'no_', 'saturated_')) else "OK"
                    print(f"      {status}: {flag.replace('_', ' ').title()}")
    
    def _determine_validation_status(self, validation_result):
        """Determine overall validation status"""
        
        if validation_result['errors']:
            return 'error'
        
        quality_flags = validation_result.get('quality_flags', {})
        
        # Critical issues
        critical_flags = ['incomplete_output', 'no_lesion_detected']
        if any(quality_flags.get(flag, False) for flag in critical_flags):
            return 'warning'
        
        # Warning issues
        warning_flags = ['very_small_lesion', 'very_large_lesion', 'many_components', 
                        'single_slice_lesion', 'spans_many_slices', 'saturated_intensity']
        if any(quality_flags.get(flag, False) for flag in warning_flags):
            return 'warning'
        
        return 'success'
    
    def validate_batch_results(self, base_dir, pattern="*ADS_Results*", verbose=False):
        """
        Validate multiple result directories
        
        Args:
            base_dir: Base directory containing result folders
            pattern: Pattern to match result folders
            verbose: Show detailed output
            
        Returns:
            List of validation results
        """
        
        print(f"Searching for results in: {base_dir}")
        print(f"   Pattern: {pattern}")
        
        # Find result directories
        search_pattern = os.path.join(base_dir, "**", pattern)
        result_dirs = glob.glob(search_pattern, recursive=True)
        result_dirs = [d for d in result_dirs if os.path.isdir(d)]
        
        print(f"Found {len(result_dirs)} result directories")
        
        validation_results = []
        
        for i, results_dir in enumerate(result_dirs, 1):
            if not verbose:
                print(f"   Processing {i}/{len(result_dirs)}: {os.path.basename(results_dir)}", end="")
            
            result = self.validate_single_result(results_dir, verbose)
            validation_results.append(result)
            
            if not verbose:
                status_emoji = {"success": "OK", "warning": "WARN", "error": "ERROR"}
                print(f" [{status_emoji.get(result['status'], 'UNKNOWN')}]")
        
        self.validation_results = validation_results
        return validation_results
    
    def generate_summary_report(self, output_dir=".", save_plots=True):
        """Generate comprehensive summary report"""
        
        if not self.validation_results:
            print("No validation results to summarize")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary statistics
        df = pd.DataFrame(self.validation_results)
        
        # Extract lesion metrics into separate columns
        lesion_metrics = []
        for result in self.validation_results:
            metrics = result.get('lesion_metrics', {})
            row = {
                'subject_id': result['subject_id'],
                'status': result['status'],
                'lesion_volume_ml': metrics.get('lesion_volume_ml', 0),
                'lesion_voxels': metrics.get('lesion_voxels', 0),
                'num_components': metrics.get('num_components', 0),
                'affected_slices': metrics.get('affected_slices', 0),
                'total_slices': metrics.get('total_slices', 0),
                'mean_intensity': metrics.get('mean_intensity', 0)
            }
            
            # Add quality flags
            quality_flags = result.get('quality_flags', {})
            for flag, value in quality_flags.items():
                row[f'flag_{flag}'] = value
            
            lesion_metrics.append(row)
        
        metrics_df = pd.DataFrame(lesion_metrics)
        
        # Save detailed CSV
        csv_path = os.path.join(output_dir, f"validation_results_{timestamp}.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        # Generate summary statistics
        print(f"\nVALIDATION SUMMARY:")
        print(f"{'='*50}")
        
        status_counts = metrics_df['status'].value_counts()
        total = len(metrics_df)
        
        for status in ['success', 'warning', 'error']:
            count = status_counts.get(status, 0)
            percentage = (count / total) * 100
            print(f"{status.title()}: {count} ({percentage:.1f}%)")
        
        # Lesion volume statistics
        successful = metrics_df[metrics_df['lesion_volume_ml'] > 0]
        if len(successful) > 0:
            print(f"\nLESION VOLUME STATISTICS:")
            print(f"   Subjects with lesions: {len(successful)}/{total}")
            print(f"   Mean volume: {successful['lesion_volume_ml'].mean():.2f} ml")
            print(f"   Median volume: {successful['lesion_volume_ml'].median():.2f} ml")
            print(f"   Range: {successful['lesion_volume_ml'].min():.2f} - {successful['lesion_volume_ml'].max():.2f} ml")
            print(f"   Standard deviation: {successful['lesion_volume_ml'].std():.2f} ml")
        
        # Quality flags summary
        flag_columns = [col for col in metrics_df.columns if col.startswith('flag_')]
        if flag_columns:
            print(f"\nQUALITY FLAGS SUMMARY:")
            for col in flag_columns:
                flag_name = col.replace('flag_', '').replace('_', ' ').title()
                count = metrics_df[col].sum()
                percentage = (count / total) * 100
                if count > 0:
                    print(f"   {flag_name}: {count} ({percentage:.1f}%)")
        
        # Generate plots if requested
        if save_plots and len(successful) > 0:
            self._generate_summary_plots(metrics_df, output_dir, timestamp)
        
        # Save text summary
        summary_path = os.path.join(output_dir, f"validation_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Validation Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"Total subjects: {total}\n")
            for status in ['success', 'warning', 'error']:
                count = status_counts.get(status, 0)
                percentage = (count / total) * 100
                f.write(f"{status.title()}: {count} ({percentage:.1f}%)\n")
            
            if len(successful) > 0:
                f.write(f"\nLesion Volume Statistics:\n")
                f.write(f"Subjects with lesions: {len(successful)}/{total}\n")
                f.write(f"Mean volume: {successful['lesion_volume_ml'].mean():.2f} ml\n")
                f.write(f"Median volume: {successful['lesion_volume_ml'].median():.2f} ml\n")
                f.write(f"Range: {successful['lesion_volume_ml'].min():.2f} - {successful['lesion_volume_ml'].max():.2f} ml\n")
                f.write(f"Standard deviation: {successful['lesion_volume_ml'].std():.2f} ml\n")
        
        print(f"Summary report saved to: {summary_path}")
        
        return metrics_df
    
    def _generate_summary_plots(self, metrics_df, output_dir, timestamp):
        """Generate summary plots"""
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ADS Validation Summary', fontsize=16, fontweight='bold')
            
            # 1. Status distribution
            status_counts = metrics_df['status'].value_counts()
            colors = {'success': 'green', 'warning': 'orange', 'error': 'red'}
            status_colors = [colors.get(status, 'gray') for status in status_counts.index]
            
            axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                          colors=status_colors, startangle=90)
            axes[0, 0].set_title('Validation Status Distribution')
            
            # 2. Lesion volume distribution
            successful = metrics_df[metrics_df['lesion_volume_ml'] > 0]
            if len(successful) > 0:
                axes[0, 1].hist(successful['lesion_volume_ml'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 1].set_xlabel('Lesion Volume (ml)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Lesion Volume Distribution')
                axes[0, 1].axvline(successful['lesion_volume_ml'].mean(), color='red', linestyle='--', 
                                  label=f'Mean: {successful["lesion_volume_ml"].mean():.2f} ml')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No lesions detected', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Lesion Volume Distribution')
            
            # 3. Number of components vs volume
            if len(successful) > 0:
                scatter = axes[1, 0].scatter(successful['lesion_volume_ml'], successful['num_components'], 
                                           alpha=0.6, c=successful['num_components'], cmap='viridis')
                axes[1, 0].set_xlabel('Lesion Volume (ml)')
                axes[1, 0].set_ylabel('Number of Components')
                axes[1, 0].set_title('Components vs Volume')
                plt.colorbar(scatter, ax=axes[1, 0], label='Components')
            else:
                axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Components vs Volume')
            
            # 4. Quality flags frequency
            flag_columns = [col for col in metrics_df.columns if col.startswith('flag_')]
            if flag_columns:
                flag_counts = {}
                for col in flag_columns:
                    flag_name = col.replace('flag_', '').replace('_', ' ').title()
                    count = metrics_df[col].sum()
                    if count > 0:
                        flag_counts[flag_name] = count
                
                if flag_counts:
                    y_pos = np.arange(len(flag_counts))
                    axes[1, 1].barh(y_pos, list(flag_counts.values()), alpha=0.7)
                    axes[1, 1].set_yticks(y_pos)
                    axes[1, 1].set_yticklabels(list(flag_counts.keys()))
                    axes[1, 1].set_xlabel('Frequency')
                    axes[1, 1].set_title('Quality Flags Frequency')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No quality flags raised', ha='center', va='center', 
                                   transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].set_title('Quality Flags Frequency')
            else:
                axes[1, 1].text(0.5, 0.5, 'No quality data', ha='center', va='center', 
                               transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Quality Flags Frequency')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"validation_summary_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Summary plots saved to: {plot_path}")
            
        except Exception as e:
            print(f"Could not generate plots: {e}")

def get_arg_parser():
    parser = argparse.ArgumentParser(
        prog='validate_results',
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Results Validation for Acute Stroke Detection & Segmentation

Comprehensive validation and analysis of ADS processing results including:
- File completeness checking
- Lesion metrics analysis  
- Quality assessment
- Statistical summaries
- Automated report generation

Examples:
  # Validate single result
  python validate_results.py /path/to/Subject01_ADS_Results_20250525_031533/

  # Validate all results in directory
  python validate_results.py /path/to/subjects/ -batch

  # Generate detailed report with plots
  python validate_results.py /path/to/subjects/ -batch -report -plots
        """
    )
    
    parser.add_argument('input_path', type=str,
                        help='Path to results directory or base directory containing multiple results')
    
    parser.add_argument('-batch', dest='batch_mode', action='store_true',
                        help='Batch mode: validate multiple result directories')
    
    parser.add_argument('-pattern', dest='pattern', type=str, default='*ADS_Results*',
                        help='Pattern to match result directories in batch mode (default: *ADS_Results*)')
    
    parser.add_argument('-report', dest='generate_report', action='store_true',
                        help='Generate comprehensive summary report')
    
    parser.add_argument('-plots', dest='generate_plots', action='store_true',
                        help='Generate summary plots (requires -report)')
    
    parser.add_argument('-output', dest='output_dir', type=str, default='.',
                        help='Output directory for reports and plots (default: current directory)')
    
    parser.add_argument('-verbose', dest='verbose', action='store_true',
                        help='Show detailed validation output')
    
    parser.add_argument('-quiet', dest='quiet', action='store_true',
                        help='Suppress non-essential output')
    
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print("\n" + "="*60)
        print("RESULTS VALIDATION - ACUTE STROKE DETECTION & SEGMENTATION")
        print("="*60)
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    # Initialize validator
    validator = ResultsValidator()
    
    try:
        if args.batch_mode:
            # Batch validation
            if not args.quiet:
                print(f"Running batch validation...")
            
            results = validator.validate_batch_results(
                args.input_path, 
                pattern=args.pattern,
                verbose=args.verbose
            )
            
            if not results:
                print("No results found to validate")
                sys.exit(1)
            
            # Print summary
            if not args.quiet:
                status_counts = {}
                for result in results:
                    status = result['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                print(f"\nBatch Validation Summary:")
                total = len(results)
                for status in ['success', 'warning', 'error']:
                    count = status_counts.get(status, 0)
                    percentage = (count / total) * 100
                    print(f"   {status.title()}: {count} ({percentage:.1f}%)")
            
            # Generate comprehensive report
            if args.generate_report:
                if not args.quiet:
                    print(f"\nGenerating comprehensive report...")
                
                os.makedirs(args.output_dir, exist_ok=True)
                validator.generate_summary_report(args.output_dir, args.generate_plots)
        
        else:
            # Single validation
            if not args.quiet:
                print(f"Validating single result: {os.path.basename(args.input_path)}")
            
            result = validator.validate_single_result(args.input_path, verbose=True)
            
            # Print result
            status_display = {"success": "SUCCESS", "warning": "WARNING", "error": "ERROR"}
            print(f"\nValidation Status: {status_display.get(result['status'], 'UNKNOWN')}")
            
            if result['errors']:
                print(f"Errors:")
                for error in result['errors']:
                    print(f"   - {error}")
            
            if result['warnings']:
                print(f"Warnings:")
                for warning in result['warnings']:
                    print(f"   - {warning}")
            
            # Show lesion metrics if available
            if 'lesion_metrics' in result:
                metrics = result['lesion_metrics']
                print(f"\nLesion Metrics:")
                print(f"   Volume: {metrics['lesion_volume_ml']:.2f} ml ({metrics['lesion_voxels']:,} voxels)")
                print(f"   Components: {metrics['num_components']}")
                print(f"   Affected slices: {metrics['affected_slices']}/{metrics['total_slices']}")
        
        if not args.quiet:
            print(f"\nValidation completed successfully!")
            
    except KeyboardInterrupt:
        print(f"\nValidation interrupted by user")
    except Exception as e:
        print(f"\nValidation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()