# Copyright (c) 2021, Andreia V. Faria, Chin-Fu Liu
# All rights reserved.
#
# This program is free software; 
# you can redistribute it and/or modify it under the terms of the GNU General Public License v3.0. 
# See the LICENSE file or read the terms at https://www.gnu.org/licenses/gpl-3.0.en.html.

"""
Performance-optimized command line interface for Acute Stroke Detection & Segmentation.
Features enhanced argument parsing, system validation, GPU auto-detection, 
progress tracking, and comprehensive error handling for clinical deployment.
"""

import sys
import os
import argparse

from ads_bin_optimizations import *

OPT_MODEL = "-model"
OPT_INPUTFOLDER = "-input"
OPT_BVALUE = "-bvalue"
OPT_SAVEMNI = "-save_MNI"
OPT_GENMASK = "-generate_brainmask"
OPT_GENREPORT = "-generate_report"
OPT_GENRESULTPNG = "-generate_result_png"
OPT_CREATERESULTSFOLDER = "-create_results_folder"
OPT_SHOWPROGRESS = "-show_progress"  # New option
OPT_CLEARMODELS = "-clear_models"    # New option

def get_arg_parser():
    parser = argparse.ArgumentParser(prog='ADSRun', formatter_class=argparse.RawTextHelpFormatter,
    description="\n This software allows creation of lesion predicts for acute ischemic subjects in biomedical NIFTI GZ volumes.\n"+\
                "The project is hosted at: https://github.com/Chin-Fu-Liu/Acute_Stroke_Detection/ \n"+\
                "See the documentation for details on its use.\n"+\
                "For questions and feedback, please contact:cliu104@jhu.edu\n\n"+\
                "PERFORMANCE OPTIMIZED VERSION\n"+\
                "Features: GPU auto-detection, model caching, progress tracking, memory optimization")
    
    parser.add_argument(OPT_INPUTFOLDER, dest='input',  type=str, 
                        help='Specify the subject input folder. FolderName should be SubjID and  the naming and format of images under subject folder should fulfill the requirements in biomedical NIFTI GZ volumes as  https://github.com/Chin-Fu-Liu/Acute_Stroke_Detection/')
    
    parser.add_argument(OPT_MODEL, dest='model', type=str, default='DAGMNet_CH3',
                        help='Specify which the trained model to be used, like DAGMNet_CH3, DAGMNet_CH2, UNet_CH3, UNet_CH2, FCN_CH3, FCN_CH2. Models should be under the Trained_Networks folder with the same naming. (default: DAGMNet_CH3)')

    parser.add_argument(OPT_BVALUE, dest='bvalue', type=int, default=1000,
                        help='Specify the b-value to calculate ADC, if ADC is not given. If ADC is given under subjectID folder, this option will be ignored. (default: 1000)')
    
    parser.add_argument(OPT_SAVEMNI, dest='save_MNI', action='store_true', default=True,
                        help='Save all images in MNI space. (default: True)')
    
    parser.add_argument('--no_save_MNI', dest='save_MNI', action='store_false',
                        help='Do not save images in MNI space.')

    parser.add_argument(OPT_GENMASK, dest='generate_brainmask', action='store_true', default=False,
                        help='Generate brain mask files. (default: False)')
    
    parser.add_argument(OPT_GENREPORT, dest='generate_report', action='store_true', default=True,
                        help='Generate lesion report in txt format. (default: True)')
    
    parser.add_argument('--no_report', dest='generate_report', action='store_false',
                        help='Do not generate lesion report.')
    
    parser.add_argument(OPT_GENRESULTPNG, dest='generate_result_png', action='store_true', default=True,
                        help='Generate result visualization in png format. (default: True)')
    
    parser.add_argument('--no_png', dest='generate_result_png', action='store_false',
                        help='Do not generate result PNG.')
    
    parser.add_argument(OPT_CREATERESULTSFOLDER, dest='create_results_folder', action='store_true', default=True,
                        help='Create a timestamped results folder to organize outputs. (default: True)')
    
    parser.add_argument('--no_results_folder', dest='create_results_folder', action='store_false',
                        help='Save results in the input folder instead of creating a new results folder.')
    
    # Performance options
    parser.add_argument(OPT_SHOWPROGRESS, dest='show_progress', action='store_true', default=False,
                        help='Show detailed progress information, memory usage, and performance metrics. (default: False)')
    
    parser.add_argument(OPT_CLEARMODELS, dest='clear_models', action='store_true', default=False,
                        help='Clear models from memory after processing (useful for batch processing). (default: False)')
    
    # Verbose option
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                        help='Enable verbose output with detailed logging.')
    
    # Quiet option
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', default=False,
                        help='Suppress non-essential output.')
    
    args = parser.parse_args()
    return args

def print_system_info():
    """Print system and environment information"""
    import platform
    import psutil
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
    
    # GPU info
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"GPU: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("GPU: No GPUs detected")
    except:
        print("GPU: Unable to detect GPU information")
    
    print("="*60)

def validate_inputs(args):
    """Validate input arguments and files"""
    errors = []
    warnings = []
    
    # Check if input folder exists
    if not os.path.exists(args.input):
        errors.append(f"Input folder does not exist: {args.input}")
        return errors, warnings
    
    # Extract subject ID
    subj_id = os.path.basename(args.input.rstrip('/'))
    
    # Check for required files
    required_files = [
        (f"{subj_id}_DWI.nii.gz", f"{subj_id}_DWI.nii"),
        (f"{subj_id}_b0.nii.gz", f"{subj_id}_b0.nii")
    ]
    
    for primary, backup in required_files:
        primary_path = os.path.join(args.input, primary)
        backup_path = os.path.join(args.input, backup)
        
        if not os.path.exists(primary_path) and not os.path.exists(backup_path):
            errors.append(f"Required file not found: {primary} or {backup}")
    
    # Check for optional ADC file
    adc_files = [f"{subj_id}_ADC.nii.gz", f"{subj_id}_ADC.nii"]
    adc_exists = any(os.path.exists(os.path.join(args.input, adc)) for adc in adc_files)
    
    if not adc_exists:
        warnings.append(f"ADC file not found - will be calculated using b-value {args.bvalue}")
    
    # Check model file
    try:
        from ads_bin_optimizations import get_DirPaths
        _, _, _, TrainedNetsDir = get_DirPaths()
        model_path = os.path.join(TrainedNetsDir, f"{args.model}.h5")
        
        if not os.path.exists(model_path):
            errors.append(f"Model file not found: {model_path}")
    except:
        warnings.append("Could not verify model file existence")
    
    return errors, warnings

def main(args):
    # Print header
    if not args.quiet:
        print("\n" + "="*60)
        print("ACUTE STROKE DETECTION & SEGMENTATION")
        print("="*60)
        print("Performance Optimized Version")
        print(f"Model: {args.model}")
        print(f"Subject: {os.path.basename(args.input.rstrip('/'))}")
    
    # Show system info if verbose or show_progress
    if args.verbose or args.show_progress:
        print_system_info()
    
    # Validate inputs
    if not args.quiet:
        print("\nValidating inputs...")
    
    errors, warnings = validate_inputs(args)
    
    if errors:
        print("\nERRORS FOUND:")
        for error in errors:
            print(f"  • {error}")
        print("\nPlease fix these errors before proceeding.")
        sys.exit(1)
    
    if warnings and not args.quiet:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not args.quiet:
        print("Input validation passed!")
    
    # Extract parameters
    SubjDir = args.input
    model_name = args.model
    bvalue = args.bvalue
    save_MNI = args.save_MNI
    generate_brainmask = args.generate_brainmask
    generate_report = args.generate_report
    generate_result_png = args.generate_result_png
    create_results_folder_flag = args.create_results_folder
    show_progress = args.show_progress or args.verbose
    clear_models_after = args.clear_models
    
    # Show configuration
    if not args.quiet:
        print(f"\nCONFIGURATION:")
        print(f"  Model: {model_name}")
        print(f"  B-value: {bvalue}")
        print(f"  Save MNI: {save_MNI}")
        print(f"  Generate mask: {generate_brainmask}")
        print(f"  Generate report: {generate_report}")
        print(f"  Generate PNG: {generate_result_png}")
        print(f"  Results folder: {create_results_folder_flag}")
        print(f"  Show progress: {show_progress}")
        print(f"  Clear models: {clear_models_after}")
    
    # Start processing
    start_time = time.time()
    
    try:
        results_dir, performance_tracker = ADS(
            SubjDir=SubjDir,
            model_name=model_name,
            bvalue=bvalue,
            save_MNI=save_MNI,
            generate_brainmask=generate_brainmask,
            generate_report=generate_report,
            generate_result_png=generate_result_png,
            create_results_folder_flag=create_results_folder_flag,
            show_progress=show_progress,
            clear_models_after=clear_models_after
        )
        
        total_time = time.time() - start_time
        
        if not args.quiet:
            print(f"\nSUCCESS! Processing completed in {total_time:.2f} seconds")
            print(f"Results saved to: {results_dir}")
            
            # Quick validation
            subj_id = os.path.basename(args.input.rstrip('/'))
            lesion_file = os.path.join(results_dir, f"{subj_id}_{model_name}_Lesion_Predict.nii.gz")
            
            if os.path.exists(lesion_file):
                import nibabel as nib
                lesion_img = nib.load(lesion_file)
                lesion_data = lesion_img.get_fdata()
                lesion_voxels = np.sum(lesion_data > 0.5)
                voxel_volume = np.prod(lesion_img.header.get_zooms()[:3])
                lesion_volume_ml = (lesion_voxels * voxel_volume) / 1000
                
                print(f"\nQUICK RESULTS:")
                print(f"  Lesion volume: {lesion_volume_ml:.2f} ml")
                print(f"  Lesion voxels: {lesion_voxels:,}")
                
                if lesion_volume_ml == 0:
                    print("  No lesion detected")
                elif lesion_volume_ml < 0.5:
                    print("  Small lesion detected")
                elif lesion_volume_ml > 100:
                    print("  Large lesion detected")
                else:
                    print("  Lesion detected")
        
    except Exception as e:
        print(f"\nERROR: Processing failed!")
        print(f"Error details: {str(e)}")
        
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":       
    args = get_arg_parser()
    
    # Handle help and no arguments
    if len(sys.argv) == 1:
        print("Acute Stroke Detection & Segmentation")
        print("For help on usage, please use the option -h or --help")
        sys.exit(1)
    
    # Validate required arguments
    if not args.input:
        print("ERROR: Option ["+OPT_INPUTFOLDER+"] must be specified.")
        print("Please try [-h] for more information.")
        sys.exit(1)
    
    # Handle conflicting options
    if args.verbose and args.quiet:
        print("Warning: Both --verbose and --quiet specified. Using verbose mode.")
        args.quiet = False
    
    main(args)