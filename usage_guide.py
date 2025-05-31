#!/usr/bin/env python3
"""
Comprehensive Usage Guide for Acute Stroke Detection & Segmentation
This script provides an interactive guide with examples, best practices, and troubleshooting
information for using the ADS system effectively in various research and clinical scenarios.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import textwrap

class ADSUsageGuide:
    """Interactive usage guide for ADS"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.examples = self._load_examples()
    
    def _load_examples(self):
        """Load usage examples"""
        return {
            "basic": {
                "title": "Basic Single Subject Processing",
                "description": "Process a single subject with default settings",
                "command": 'python codes/ADSRun.py -input "data/examples/Subject01/"',
                "explanation": """
                This is the simplest way to process a stroke subject:
                - Uses default DAGMNet_CH3 model
                - Creates timestamped results folder
                - Generates all standard outputs (lesion mask, report, visualization)
                - Saves MNI-normalized images
                """,
                "requirements": [
                    "SubjectID_DWI.nii.gz (required)",
                    "SubjectID_b0.nii.gz (required)", 
                    "SubjectID_ADC.nii.gz (optional - will be calculated)"
                ]
            },
            
            "advanced": {
                "title": "Advanced Single Subject Processing",
                "description": "Process with custom settings and performance monitoring",
                "command": '''python codes/ads_run_optimizations.py \\
    -input "data/examples/Subject01/" \\
    -model UNet_CH3 \\
    -bvalue 800 \\
    -generate_brainmask \\
    -show_progress \\
    --verbose''',
                "explanation": """
                Advanced processing with customization:
                - Uses UNet_CH3 model instead of default
                - Custom b-value for ADC calculation
                - Generates brain mask files
                - Shows detailed progress and performance metrics
                - Verbose output with system information
                """,
                "requirements": [
                    "Same as basic example",
                    "Optional: Different b-value if ADC not provided"
                ]
            },
            
            "minimal": {
                "title": "Minimal Output Processing",
                "description": "Process with minimal outputs for faster processing",
                "command": '''python codes/ads_run_optimizations.py \\
    -input "data/examples/Subject01/" \\
    --no_save_MNI \\
    --no_report \\
    --no_png \\
    --no_results_folder \\
    --quiet''',
                "explanation": """
                Minimal processing for speed:
                - Only generates essential lesion mask
                - No MNI space images (faster)
                - No detailed report
                - No visualization PNG
                - Saves directly in input folder
                - Quiet output (minimal messages)
                """,
                "requirements": [
                    "Same as basic example"
                ]
            },
            
            "batch_sequential": {
                "title": "Batch Processing (Sequential)",
                "description": "Process multiple subjects one after another",
                "command": '''python batch_process_ads.py \\
    /path/to/subjects/ \\
    -model DAGMNet_CH3 \\
    -verbose''',
                "explanation": """
                Sequential batch processing:
                - Processes all subjects in directory
                - One subject at a time (reliable)
                - Progress tracking and error handling
                - Generates comprehensive summary report
                - Automatic results organization
                """,
                "requirements": [
                    "Directory with multiple subject folders",
                    "Each subject folder with required files"
                ]
            },
            
            "batch_parallel": {
                "title": "Batch Processing (Parallel)",
                "description": "Process multiple subjects in parallel for speed",
                "command": '''python batch_process_ads.py \\
    /path/to/subjects/ \\
    -model DAGMNet_CH3 \\
    -parallel \\
    -workers 4 \\
    -verbose''',
                "explanation": """
                Parallel batch processing:
                - Processes multiple subjects simultaneously
                - Faster for large datasets
                - Configurable number of workers
                - Requires more memory and CPU cores
                - Same error handling and reporting as sequential
                """,
                "requirements": [
                    "Same as sequential batch",
                    "Sufficient system resources (RAM, CPU cores)"
                ]
            },
            
            "validation_single": {
                "title": "Single Result Validation",
                "description": "Validate and analyze a single processing result",
                "command": '''python validate_results.py \\
    "data/examples/Subject01/Subject01_ADS_Results_20250525_031533/" \\
    -verbose''',
                "explanation": """
                Detailed validation of processing results:
                - Checks file completeness
                - Analyzes lesion metrics (volume, components, distribution)
                - Performs quality checks and flags potential issues
                - Provides detailed quantitative analysis
                """,
                "requirements": [
                    "Completed ADS processing results folder"
                ]
            },
            
            "validation_batch": {
                "title": "Batch Result Validation",
                "description": "Validate multiple results and generate summary reports",
                "command": '''python validate_results.py \\
    /path/to/subjects/ \\
    -batch \\
    -report \\
    -plots \\
    -output validation_reports/''',
                "explanation": """
                Comprehensive batch validation:
                - Validates all results in directory
                - Generates statistical summary
                - Creates visualization plots
                - Exports detailed CSV reports
                - Quality assessment across entire dataset
                """,
                "requirements": [
                    "Directory containing multiple ADS result folders"
                ]
            },
            
            "clinical_workflow": {
                "title": "Clinical Research Workflow",
                "description": "Complete workflow for clinical research study",
                "command": '''# 1. Process all subjects
python batch_process_ads.py /path/to/study_data/ -model DAGMNet_CH3 -verbose

# 2. Validate all results  
python validate_results.py /path/to/study_data/ -batch -report -plots

# 3. Generate final summary
python -c "
import pandas as pd
df = pd.read_csv('validation_results_*.csv')
print('Study Summary:')
print(f'Total subjects: {len(df)}')
print(f'Successful: {len(df[df.status == \"success\"])}')
print(f'Mean lesion volume: {df[df.lesion_volume_ml > 0].lesion_volume_ml.mean():.2f} ml')
"''',
                "explanation": """
                Complete clinical research workflow:
                - Batch processing with error handling
                - Comprehensive result validation
                - Statistical analysis and reporting
                - Quality assurance throughout
                """,
                "requirements": [
                    "Properly organized study data",
                    "Validated ADS environment",
                    "Sufficient computational resources"
                ]
            }
        }
    
    def show_example(self, example_key, detailed=False):
        """Show a specific usage example"""
        if example_key not in self.examples:
            print(f"Example '{example_key}' not found")
            return
        
        example = self.examples[example_key]
        
        print(f"\n{'='*60}")
        print(f"{example['title']}")
        print(f"{'='*60}")
        print(f"{example['description']}")
        
        print(f"\nCOMMAND:")
        print("-" * 40)
        print(example['command'])
        
        if detailed:
            print(f"\nEXPLANATION:")
            print("-" * 40)
            print(textwrap.dedent(example['explanation']).strip())
            
            print(f"\nREQUIREMENTS:")
            print("-" * 40)
            for req in example['requirements']:
                print(f"  - {req}")
    
    def show_all_examples(self, detailed=False):
        """Show all usage examples"""
        print("\n" + "="*60)
        print("ADS USAGE GUIDE - ALL EXAMPLES")
        print("="*60)
        
        for i, (key, example) in enumerate(self.examples.items(), 1):
            print(f"\n{i}. {example['title']}")
            print(f"   {example['description']}")
            if not detailed:
                print(f"   Command: {example['command'].split()[0]} ...")
        
        if not detailed:
            print(f"\nUse --detailed flag to see full explanations and requirements")
        else:
            for key in self.examples.keys():
                self.show_example(key, detailed=True)
    
    def show_quick_start(self):
        """Show quick start guide"""
        print("\n" + "="*60)
        print("ADS QUICK START GUIDE")
        print("="*60)
        
        steps = [
            {
                "title": "1. Prepare Your Data", 
                "commands": [
                    "mkdir -p data/examples/MySubject/",
                    "# Copy your files:",
                    "#   MySubject_DWI.nii.gz",
                    "#   MySubject_b0.nii.gz"
                ],
                "description": "Organize your imaging data in the correct structure"
            },
            {
                "title": "2. Process Single Subject",
                "commands": [
                    'python codes/ADSRun.py -input "data/examples/MySubject/"'
                ],
                "description": "Run stroke detection and segmentation"
            },
            {
                "title": "3. Validate Results",
                "commands": [
                    'python validate_results.py "data/examples/MySubject/MySubject_ADS_Results_*/" -verbose'
                ],
                "description": "Check processing quality and get detailed metrics"
            },
            {
                "title": "4. Process Multiple Subjects (Optional)",
                "commands": [
                    'python batch_process_ads.py "data/examples/" -verbose'
                ],
                "description": "For studies with multiple subjects"
            }
        ]
        
        for step in steps:
            print(f"\n{step['title']}")
            print("-" * 50)
            print(f"   {step['description']}")
            print()
            for cmd in step['commands']:
                if cmd.startswith('#'):
                    print(f"   {cmd}")
                else:
                    print(f"   $ {cmd}")
    
    def show_troubleshooting(self):
        """Show troubleshooting guide"""
        print("\n" + "="*60)
        print("ADS TROUBLESHOOTING GUIDE")
        print("="*60)
        
        issues = [
            {
                "problem": "Import Error: Cannot import ads_bin_optimizations",
                "solution": [
                    "Check that you're running from the project root directory",
                    "Verify ads_bin_optimizations.py exists",
                    "Check Python path configuration"
                ]
            },
            {
                "problem": "Missing Model Files",
                "solution": [
                    "Download models from: https://www.nitrc.org/projects/ads",
                    "Place .h5 files in data/Trained_Nets/",
                    "Verify file paths and permissions"
                ]
            },
            {
                "problem": "GPU Not Detected",
                "solution": [
                    "Check CUDA installation",
                    "Install tensorflow-gpu if needed",
                    "ADS will automatically fall back to CPU"
                ]
            },
            {
                "problem": "Processing Fails with Memory Error",
                "solution": [
                    "Close other applications",
                    "Use --no_save_MNI to reduce memory usage", 
                    "For batch processing, reduce number of workers",
                    "Process subjects individually if needed"
                ]
            },
            {
                "problem": "No Lesion Detected",
                "solution": [
                    "Check input image quality",
                    "Verify correct b-value setting",
                    "Try different models (UNet_CH3, DAGMNet_CH2)",
                    "Check if stroke is actually present"
                ]
            },
            {
                "problem": "Template Processing Error",
                "solution": [
                    "Download template files from original repository",
                    "Verify file integrity",
                    "Check file permissions"
                ]
            }
        ]
        
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['problem']}")
            print("   Solutions:")
            for solution in issue['solution']:
                print(f"   - {solution}")
    
    def show_best_practices(self):
        """Show best practices guide"""
        print("\n" + "="*60)
        print("ADS BEST PRACTICES")
        print("="*60)
        
        practices = [
            {
                "category": "Data Organization",
                "tips": [
                    "Use consistent naming: SubjectID_DWI.nii.gz, SubjectID_b0.nii.gz",
                    "Keep original data separate from processed results",
                    "Use descriptive subject IDs (avoid spaces and special characters)",
                    "Organize by study/cohort for large datasets"
                ]
            },
            {
                "category": "Model Selection",
                "tips": [
                    "DAGMNet_CH3: Best overall performance (recommended)",
                    "DAGMNet_CH2: Faster processing, good for large studies",
                    "UNet_CH3: Good alternative if DAGMNet fails",
                    "Test different models on pilot data to choose optimal"
                ]
            },
            {
                "category": "Performance Optimization",
                "tips": [
                    "Use GPU when available (3-5x faster)",
                    "For batch processing: balance workers vs available RAM",
                    "Use --no_save_MNI for faster processing if MNI images not needed",
                    "Clear models between subjects in batch processing"
                ]
            },
            {
                "category": "Quality Control",
                "tips": [
                    "Always validate results with validate_results.py",
                    "Manually review cases with warnings",
                    "Check for artifacts in very large or very small lesions",
                    "Compare results across different models for critical cases"
                ]
            },
            {
                "category": "Clinical Research",
                "tips": [
                    "Document processing parameters for reproducibility",
                    "Keep processing logs for audit trails", 
                    "Validate environment before large studies",
                    "Have manual review protocol for automated results"
                ]
            }
        ]
        
        for practice in practices:
            print(f"\n{practice['category']}")
            print("-" * 50)
            for tip in practice['tips']:
                print(f"   - {tip}")
    
    def interactive_guide(self):
        """Run interactive guide"""
        print("\n" + "="*60)
        print("ADS INTERACTIVE GUIDE")
        print("="*60)
        
        while True:
            print(f"\nWhat would you like to know about?")
            print("1. Quick Start Guide")
            print("2. Usage Examples")
            print("3. Troubleshooting")
            print("4. Best Practices")
            print("5. Specific Example (by name)")
            print("6. Exit")
            
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.show_quick_start()
                elif choice == '2':
                    detailed = input("Show detailed explanations? (y/N): ").lower().startswith('y')
                    self.show_all_examples(detailed=detailed)
                elif choice == '3':
                    self.show_troubleshooting()
                elif choice == '4':
                    self.show_best_practices()
                elif choice == '5':
                    print("\nAvailable examples:")
                    for key, example in self.examples.items():
                        print(f"  - {key}: {example['title']}")
                    example_name = input("\nEnter example name: ").strip()
                    self.show_example(example_name, detailed=True)
                elif choice == '6':
                    print("Thanks for using ADS! Good luck with your stroke research!")
                    break
                else:
                    print("Invalid choice. Please enter 1-6.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

def main():
    parser = argparse.ArgumentParser(
        prog='usage_guide',
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Comprehensive Usage Guide for Acute Stroke Detection & Segmentation

This interactive guide provides examples, best practices, and troubleshooting
for using the ADS system effectively.

Examples:
  # Interactive guide
  python usage_guide.py --interactive

  # Show all examples  
  python usage_guide.py --examples

  # Show specific example
  python usage_guide.py --example basic

  # Quick start guide
  python usage_guide.py --quick-start
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive guide')
    
    parser.add_argument('--examples', action='store_true',
                        help='Show all usage examples')
    
    parser.add_argument('--example', type=str,
                        help='Show specific example by name')
    
    parser.add_argument('--quick-start', action='store_true',
                        help='Show quick start guide')
    
    parser.add_argument('--troubleshooting', action='store_true',
                        help='Show troubleshooting guide')
    
    parser.add_argument('--best-practices', action='store_true',
                        help='Show best practices guide')
    
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed explanations')
    
    parser.add_argument('--list-examples', action='store_true',
                        help='List available example names')
    
    args = parser.parse_args()
    
    # Initialize guide
    guide = ADSUsageGuide()
    
    try:
        if args.interactive:
            guide.interactive_guide()
        elif args.examples:
            guide.show_all_examples(detailed=args.detailed)
        elif args.example:
            guide.show_example(args.example, detailed=True)
        elif args.quick_start:
            guide.show_quick_start()
        elif args.troubleshooting:
            guide.show_troubleshooting()
        elif args.best_practices:
            guide.show_best_practices()
        elif args.list_examples:
            print("Available Examples:")
            for key, example in guide.examples.items():
                print(f"  - {key:<20} - {example['title']}")
        else:
            # Default: show quick start
            guide.show_quick_start()
            print(f"\nUse --help to see all available options")
            print(f"Use --interactive for the full interactive experience")
            
    except KeyboardInterrupt:
        print(f"\nGuide interrupted. Use --help for usage options.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()