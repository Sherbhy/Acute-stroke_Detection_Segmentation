#!/usr/bin/env python3
"""
Batch Processing Script for Acute Stroke Detection & Segmentation
This script processes multiple subjects automatically with progress tracking, 
error handling, and options for both sequential and parallel processing.
"""

import os
import sys
import glob
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# Add the codes directory to path to import ADS_bin
sys.path.append(os.path.join(os.path.dirname(__file__), 'codes'))

try:
    from ads_bin_optimizations import ADS, GPUManager, ModelManager, PerformanceTracker
except ImportError:
    print("Error: Could not import ads_bin_optimizations. Make sure you're running from the project root directory.")
    sys.exit(1)

class BatchProcessor:
    """Handle batch processing of multiple subjects"""
    
    def __init__(self):
        self.results = []
        self.successful = []
        self.failed = []
        self.start_time = None
        
    def find_subjects(self, base_dir, pattern="*"):
        """
        Find all subject directories in base directory
        
        Args:
            base_dir: Base directory containing subject folders
            pattern: Pattern to match subject folders (default: *)
        
        Returns:
            List of valid subject directories
        """
        
        print(f"Searching for subjects in: {base_dir}")
        print(f"   Pattern: {pattern}")
        
        subject_dirs = []
        
        # Find directories matching pattern
        search_pattern = os.path.join(base_dir, pattern)
        potential_dirs = glob.glob(search_pattern)
        
        for item_path in potential_dirs:
            if os.path.isdir(item_path):
                subj_id = os.path.basename(item_path)
                
                # Check for required files
                required_files = [
                    (f"{subj_id}_DWI.nii.gz", f"{subj_id}_DWI.nii"),
                    (f"{subj_id}_b0.nii.gz", f"{subj_id}_b0.nii")
                ]
                
                valid = True
                missing_files = []
                
                for primary, backup in required_files:
                    primary_path = os.path.join(item_path, primary)
                    backup_path = os.path.join(item_path, backup)
                    
                    if not os.path.exists(primary_path) and not os.path.exists(backup_path):
                        valid = False
                        missing_files.append(f"{primary} or {backup}")
                
                if valid:
                    subject_dirs.append(item_path)
                else:
                    print(f"   Skipping {subj_id}: Missing {', '.join(missing_files)}")
        
        print(f"Found {len(subject_dirs)} valid subjects:")
        for subj_dir in subject_dirs:
            print(f"   - {os.path.basename(subj_dir)}")
        
        return subject_dirs
    
    def process_single_subject(self, args):
        """
        Process a single subject (used for multiprocessing)
        
        Args:
            args: Tuple of (subj_dir, model_name, options_dict)
        
        Returns:
            Dict with processing results
        """
        
        subj_dir, model_name, options = args
        subj_id = os.path.basename(subj_dir)
        
        start_time = time.time()
        
        try:
            results_dir, performance_tracker = ADS(
                SubjDir=subj_dir,
                model_name=model_name,
                bvalue=options.get('bvalue', 1000),
                save_MNI=options.get('save_MNI', True),
                generate_brainmask=options.get('generate_brainmask', False),
                generate_report=options.get('generate_report', True),
                generate_result_png=options.get('generate_result_png', True),
                create_results_folder_flag=options.get('create_results_folder', True),
                show_progress=options.get('show_progress', False),
                clear_models_after=options.get('clear_models', True)
            )
            
            processing_time = time.time() - start_time
            
            # Get lesion volume if possible
            lesion_volume = 0
            try:
                import nibabel as nib
                import numpy as np
                
                lesion_file = os.path.join(results_dir, f"{subj_id}_{model_name}_Lesion_Predict.nii.gz")
                if os.path.exists(lesion_file):
                    lesion_img = nib.load(lesion_file)
                    lesion_data = lesion_img.get_fdata()
                    lesion_voxels = np.sum(lesion_data > 0.5)
                    voxel_volume = np.prod(lesion_img.header.get_zooms()[:3])
                    lesion_volume = (lesion_voxels * voxel_volume) / 1000  # ml
            except:
                pass
            
            return {
                'subject_id': subj_id,
                'status': 'success',
                'processing_time': processing_time,
                'results_dir': results_dir,
                'lesion_volume_ml': lesion_volume,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'subject_id': subj_id,
                'status': 'failed',
                'processing_time': processing_time,
                'results_dir': None,
                'lesion_volume_ml': 0,
                'error': str(e)
            }
    
    def process_sequential(self, subject_dirs, model_name='DAGMNet_CH3', **options):
        """
        Process subjects sequentially with progress tracking
        
        Args:
            subject_dirs: List of subject directories
            model_name: Model to use for processing
            **options: Additional processing options
        """
        
        print(f"\nStarting sequential processing of {len(subject_dirs)} subjects")
        print(f"   Model: {model_name}")
        
        self.start_time = time.time()
        
        # Configure GPU once
        GPUManager.configure_gpu()
        
        with tqdm(total=len(subject_dirs), desc="Processing subjects") as pbar:
            for i, subj_dir in enumerate(subject_dirs, 1):
                subj_id = os.path.basename(subj_dir)
                
                pbar.set_description(f"Processing {subj_id}")
                
                result = self.process_single_subject((subj_dir, model_name, options))
                self.results.append(result)
                
                if result['status'] == 'success':
                    self.successful.append(result['subject_id'])
                    pbar.set_postfix({
                        'Success': len(self.successful),
                        'Failed': len(self.failed),
                        'Time': f"{result['processing_time']:.1f}s",
                        'Volume': f"{result['lesion_volume_ml']:.1f}ml"
                    })
                else:
                    self.failed.append((result['subject_id'], result['error']))
                    pbar.set_postfix({
                        'Success': len(self.successful),
                        'Failed': len(self.failed),
                        'Error': result['error'][:30]
                    })
                
                pbar.update(1)
        
        # Clear models after all processing
        model_manager = ModelManager()
        model_manager.clear_models()
        
        self._print_summary()
        return self.results
    
    def process_parallel(self, subject_dirs, model_name='DAGMNet_CH3', max_workers=2, **options):
        """
        Process subjects in parallel
        
        Args:
            subject_dirs: List of subject directories
            model_name: Model to use for processing
            max_workers: Maximum number of parallel workers
            **options: Additional processing options
        """
        
        print(f"\nStarting parallel processing of {len(subject_dirs)} subjects")
        print(f"   Model: {model_name}")
        print(f"   Workers: {max_workers}")
        
        self.start_time = time.time()
        
        # Prepare arguments for parallel processing
        args_list = [(subj_dir, model_name, options) for subj_dir in subject_dirs]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_args = {executor.submit(self.process_single_subject, args): args for args in args_list}
            
            # Process completed jobs with progress bar
            with tqdm(total=len(subject_dirs), desc="Processing subjects") as pbar:
                for future in as_completed(future_to_args):
                    result = future.result()
                    self.results.append(result)
                    
                    if result['status'] == 'success':
                        self.successful.append(result['subject_id'])
                    else:
                        self.failed.append((result['subject_id'], result['error']))
                    
                    pbar.set_postfix({
                        'Success': len(self.successful),
                        'Failed': len(self.failed)
                    })
                    pbar.update(1)
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print processing summary"""
        
        total_time = time.time() - self.start_time
        avg_time = total_time / len(self.results) if self.results else 0
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {len(self.successful)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Average time per subject: {avg_time:.1f}s")
        
        if self.successful:
            print(f"\nSuccessfully processed:")
            for subj in self.successful:
                print(f"   - {subj}")
        
        if self.failed:
            print(f"\nFailed subjects:")
            for subj, error in self.failed:
                print(f"   - {subj}: {error}")
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save detailed results to CSV and text files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV results
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = f"batch_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nDetailed results saved to: {csv_path}")
            
            # Print statistics
            print(f"\nStatistics:")
            successful_results = df[df['status'] == 'success']
            if len(successful_results) > 0:
                print(f"   Average processing time: {successful_results['processing_time'].mean():.1f}s")
                print(f"   Average lesion volume: {successful_results['lesion_volume_ml'].mean():.2f}ml")
                print(f"   Lesion volume range: {successful_results['lesion_volume_ml'].min():.2f} - {successful_results['lesion_volume_ml'].max():.2f}ml")
        
        # Save summary text file
        summary_path = f"batch_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Batch Processing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total subjects processed: {len(self.results)}\n")
            f.write(f"Successful: {len(self.successful)}\n")
            f.write(f"Failed: {len(self.failed)}\n\n")
            
            if self.successful:
                f.write("Successful subjects:\n")
                for subj in self.successful:
                    f.write(f"  - {subj}\n")
                f.write("\n")
            
            if self.failed:
                f.write("Failed subjects:\n")
                for subj, error in self.failed:
                    f.write(f"  - {subj}: {error}\n")
        
        print(f"Summary saved to: {summary_path}")

def get_arg_parser():
    parser = argparse.ArgumentParser(
        prog='batch_process_ads',
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Batch Processing for Acute Stroke Detection & Segmentation

Process multiple subjects automatically with options for:
- Sequential or parallel processing  
- Progress tracking and error handling
- Detailed results reporting
- Automatic results organization

Examples:
  # Process all subjects in a directory
  python batch_process_ads.py /path/to/subjects/

  # Use specific model with parallel processing
  python batch_process_ads.py /path/to/subjects/ -model UNet_CH3 -parallel -workers 4

  # Process subjects matching pattern
  python batch_process_ads.py /path/to/subjects/ -pattern "Subject*" -verbose
        """
    )
    
    parser.add_argument('input_dir', type=str,
                        help='Directory containing subject folders')
    
    parser.add_argument('-model', dest='model', type=str, default='DAGMNet_CH3',
                        help='Model to use: DAGMNet_CH3, DAGMNet_CH2, UNet_CH3, UNet_CH2, FCN_CH3, FCN_CH2 (default: DAGMNet_CH3)')
    
    parser.add_argument('-pattern', dest='pattern', type=str, default='*',
                        help='Pattern to match subject folders (default: *)')
    
    parser.add_argument('-parallel', dest='parallel', action='store_true',
                        help='Use parallel processing')
    
    parser.add_argument('-workers', dest='workers', type=int, default=2,
                        help='Number of parallel workers (default: 2)')
    
    parser.add_argument('-bvalue', dest='bvalue', type=int, default=1000,
                        help='B-value for ADC calculation (default: 1000)')
    
    parser.add_argument('--no_mni', dest='save_MNI', action='store_false', default=True,
                        help='Do not save MNI space images')
    
    parser.add_argument('-generate_mask', dest='generate_brainmask', action='store_true',
                        help='Generate brain mask files')
    
    parser.add_argument('--no_report', dest='generate_report', action='store_false', default=True,
                        help='Do not generate lesion reports')
    
    parser.add_argument('--no_png', dest='generate_result_png', action='store_false', default=True,
                        help='Do not generate result PNGs')
    
    parser.add_argument('--no_results_folder', dest='create_results_folder', action='store_false', default=True,
                        help='Save results in input folders instead of creating new folders')
    
    parser.add_argument('-verbose', dest='verbose', action='store_true',
                        help='Show detailed progress and system information')
    
    parser.add_argument('-dry_run', dest='dry_run', action='store_true',
                        help='Show what would be processed without actually processing')
    
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("BATCH PROCESSING - ACUTE STROKE DETECTION & SEGMENTATION")
    print("="*60)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Initialize batch processor
    processor = BatchProcessor()
    
    # Find subjects
    subject_dirs = processor.find_subjects(args.input_dir, args.pattern)
    
    if not subject_dirs:
        print("No valid subjects found!")
        sys.exit(1)
    
    # Show configuration
    print(f"\nCONFIGURATION:")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Pattern: {args.pattern}")
    print(f"   Model: {args.model}")
    print(f"   Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        print(f"   Workers: {args.workers}")
    print(f"   B-value: {args.bvalue}")
    print(f"   Save MNI: {args.save_MNI}")
    print(f"   Generate mask: {args.generate_brainmask}")
    print(f"   Generate report: {args.generate_report}")
    print(f"   Generate PNG: {args.generate_result_png}")
    print(f"   Results folders: {args.create_results_folder}")
    
    # Dry run mode
    if args.dry_run:
        print(f"\nDRY RUN MODE - Would process {len(subject_dirs)} subjects:")
        for subj_dir in subject_dirs:
            print(f"   - {os.path.basename(subj_dir)}")
        print("\nUse without -dry_run to actually process.")
        return
    
    # Confirm processing
    print(f"\nProcess {len(subject_dirs)} subjects? (y/N): ", end="")
    if input().lower() not in ['y', 'yes']:
        print("Processing cancelled.")
        return
    
    # Show system info if verbose
    if args.verbose:
        import platform
        import psutil
        print(f"\nSYSTEM INFO:")
        print(f"   OS: {platform.system()} {platform.release()}")
        print(f"   CPU: {psutil.cpu_count(logical=True)} cores")
        print(f"   RAM: {psutil.virtual_memory().total // (1024**3)} GB")
        
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(f"   GPU: {len(gpus)} detected")
        except:
            print(f"   GPU: Unable to detect")
    
    # Prepare options
    options = {
        'bvalue': args.bvalue,
        'save_MNI': args.save_MNI,
        'generate_brainmask': args.generate_brainmask,
        'generate_report': args.generate_report,
        'generate_result_png': args.generate_result_png,
        'create_results_folder': args.create_results_folder,
        'show_progress': args.verbose,
        'clear_models': True
    }
    
    # Process subjects
    try:
        if args.parallel:
            # Limit workers based on system resources
            max_workers = min(args.workers, mp.cpu_count(), len(subject_dirs))
            results = processor.process_parallel(
                subject_dirs, 
                model_name=args.model,
                max_workers=max_workers,
                **options
            )
        else:
            results = processor.process_sequential(
                subject_dirs,
                model_name=args.model,
                **options
            )
        
        # Final message
        if len(processor.successful) == len(subject_dirs):
            print(f"\nALL SUBJECTS PROCESSED SUCCESSFULLY!")
        elif len(processor.successful) > 0:
            print(f"\n{len(processor.successful)}/{len(subject_dirs)} subjects processed successfully")
        else:
            print(f"\nNo subjects processed successfully")
            
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user")
        print(f"Processed {len(processor.results)} subjects before interruption")
    except Exception as e:
        print(f"\nBatch processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()