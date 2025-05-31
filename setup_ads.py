#!/usr/bin/env python3
"""
Setup Script for Acute Stroke Detection & Segmentation Environment
Automatically sets up and validates the ADS environment
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import json

class ADSSetup:
    """Setup and validation for ADS environment"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.requirements_installed = False
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 7):
            print(f"❌ Python {self.python_version} is not supported. Please use Python 3.7 or higher.")
            return False
        elif sys.version_info >= (3, 11):
            print(f"⚠️  Python {self.python_version} detected. Some packages may have compatibility issues.")
            print("   Recommended: Python 3.8-3.10")
        else:
            print(f"✅ Python {self.python_version} is compatible")
        
        return True
    
    def check_directory_structure(self):
        """Check and create required directory structure"""
        print("\n📁 Checking directory structure...")
        
        required_dirs = [
            "codes",
            "data",
            "data/template", 
            "data/Trained_Nets",
            "data/examples"
        ]
        
        created_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
                print(f"   📂 Created: {dir_path}")
            else:
                print(f"   ✅ Exists: {dir_path}")
        
        if created_dirs:
            print(f"   📝 Created {len(created_dirs)} missing directories")
        
        return True
    
    def check_required_files(self):
        """Check for required Python files"""
        print("\n📄 Checking required files...")
        
        required_files = [
            "codes/ADS_bin.py",
            "codes/ADSRun.py",
            "batch_process_ads.py",
            "validate_results.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                print(f"   ❌ Missing: {file_path}")
            else:
                print(f"   ✅ Present: {file_path}")
        
        if missing_files:
            print(f"\n⚠️  {len(missing_files)} required files are missing!")
            print("   Please ensure all Python files are in the correct locations.")
            return False
        
        return True
    
    def create_requirements_file(self):
        """Create requirements.txt if it doesn't exist"""
        print("\n📋 Setting up requirements...")
        
        requirements_content = """# Core dependencies
tensorflow>=2.5.0,<2.13.0
numpy>=1.19.0
nibabel>=3.2.0
scipy>=1.7.0
scikit-image>=0.18.0
matplotlib>=3.3.0
pandas>=1.3.0
tqdm>=4.60.0

# Medical imaging
dipy>=1.4.0

# Performance monitoring
psutil>=5.8.0

# Plotting and visualization
seaborn>=0.11.0

# Optional dependencies
# fury  # For DIPY visualization (optional)
# itk   # For advanced image processing (optional)
"""
        
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            with open(req_file, 'w') as f:
                f.write(requirements_content)
            print("   📝 Created requirements.txt")
        else:
            print("   ✅ requirements.txt already exists")
        
        return req_file
    
    def install_requirements(self, force=False):
        """Install Python requirements"""
        print("\n📦 Installing Python packages...")
        
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            req_file = self.create_requirements_file()
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            if force:
                cmd.append("--force-reinstall")
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ All packages installed successfully")
                self.requirements_installed = True
                return True
            else:
                print(f"   ❌ Package installation failed:")
                print(f"   {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error installing packages: {e}")
            return False
    
    def check_tensorflow_gpu(self):
        """Check TensorFlow GPU availability"""
        print("\n🎮 Checking TensorFlow GPU support...")
        
        try:
            import tensorflow as tf
            
            print(f"   TensorFlow version: {tf.__version__}")
            
            # Check for GPUs
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"   ✅ {len(gpus)} GPU(s) detected:")
                for i, gpu in enumerate(gpus):
                    print(f"      GPU {i}: {gpu.name}")
                
                # Test GPU functionality
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                        result = tf.linalg.matmul(test_tensor, test_tensor)
                    print("   ✅ GPU computation test passed")
                except Exception as e:
                    print(f"   ⚠️  GPU test failed: {e}")
            else:
                print("   ⚠️  No GPUs detected - will use CPU")
                print("   💡 For GPU support, install tensorflow-gpu or ensure CUDA is properly configured")
            
            return True
            
        except ImportError:
            print("   ❌ TensorFlow not installed")
            return False
        except Exception as e:
            print(f"   ❌ TensorFlow check failed: {e}")
            return False
    
    def check_trained_models(self):
        """Check for trained model files"""
        print("\n🤖 Checking trained models...")
        
        model_dir = self.project_root / "data" / "Trained_Nets"
        expected_models = [
            "BrainMaskNet.h5",
            "DAGMNet_CH3.h5",
            "DAGMNet_CH2.h5",
            "UNet_CH3.h5",
            "UNet_CH2.h5",
            "FCN_CH3.h5",
            "FCN_CH2.h5"
        ]
        
        missing_models = []
        present_models = []
        
        for model_name in expected_models:
            model_path = model_dir / model_name
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                present_models.append((model_name, size_mb))
                print(f"   ✅ {model_name} ({size_mb:.1f} MB)")
            else:
                missing_models.append(model_name)
                print(f"   ❌ Missing: {model_name}")
        
        if missing_models:
            print(f"\n⚠️  {len(missing_models)} model files are missing!")
            print("   These models need to be downloaded from the original repository:")
            print("   https://www.nitrc.org/projects/ads")
            print("   https://zenodo.org/record/5579390")
            return False
        
        print(f"   ✅ All {len(present_models)} models present")
        return True
    
    def check_template_files(self):
        """Check for template files"""
        print("\n🗺️  Checking template files...")
        
        template_dir = self.project_root / "data" / "template"
        expected_templates = [
            "JHU_SS_b0_padding.nii.gz",
            "JHU_SS_b0_ss_padding.nii.gz",
            "ArterialAtlas_padding.nii.gz",
            "lobe_atlas_padding.nii.gz",
            "ArterialAtlasLables.txt",
            "LobesLabelLookupTable.txt",
            "normal_mu_dwi_Res_ss_MNI_scaled_normalized.nii.gz",
            "normal_std_dwi_Res_ss_MNI_scaled_normalized.nii.gz",
            "normal_mu_ADC_Res_ss_MNI_normalized.nii.gz",
            "normal_std_ADC_Res_ss_MNI_normalized.nii.gz"
        ]
        
        missing_templates = []
        present_templates = []
        
        for template_name in expected_templates:
            template_path = template_dir / template_name
            if template_path.exists():
                if template_name.endswith('.nii.gz'):
                    size_mb = template_path.stat().st_size / (1024 * 1024)
                    present_templates.append((template_name, size_mb))
                    print(f"   ✅ {template_name} ({size_mb:.1f} MB)")
                else:
                    size_kb = template_path.stat().st_size / 1024
                    present_templates.append((template_name, size_kb))
                    print(f"   ✅ {template_name} ({size_kb:.1f} KB)")
            else:
                missing_templates.append(template_name)
                print(f"   ❌ Missing: {template_name}")
        
        if missing_templates:
            print(f"\n⚠️  {len(missing_templates)} template files are missing!")
            print("   These templates need to be downloaded from the original repository.")
            return False
        
        print(f"   ✅ All {len(present_templates)} templates present")
        return True
    
    def test_basic_functionality(self):
        """Test basic ADS functionality"""
        print("\n🧪 Testing basic functionality...")
        
        try:
            # Test imports
            sys.path.append(str(self.project_root / "codes"))
            from ADS_bin import get_DirPaths, GPUManager
            
            print("   ✅ ADS_bin imports successful")
            
            # Test directory paths
            CodesDir, ProjectDir, TemplateDir, TrainedNetsDir = get_DirPaths()
            print(f"   ✅ Directory paths resolved")
            
            # Test GPU manager
            gpu_available = GPUManager.configure_gpu()
            if gpu_available:
                print("   ✅ GPU configuration successful")
            else:
                print("   ✅ CPU configuration successful")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Functionality test failed: {e}")
            return False
    
    def create_example_data(self):
        """Create example data structure"""
        print("\n📝 Setting up example data structure...")
        
        example_dir = self.project_root / "data" / "examples" / "Subject01"
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder files (for structure demonstration)
        placeholder_files = [
            "Subject01_DWI.nii.gz",
            "Subject01_b0.nii.gz", 
            "Subject01_ADC.nii.gz"
        ]
        
        readme_content = """# Example Data Directory

This directory should contain your subject data in the following format:

## Required Files:
- SubjectID_DWI.nii.gz    # Diffusion weighted image
- SubjectID_b0.nii.gz     # B0 image (no diffusion)

## Optional Files:
- SubjectID_ADC.nii.gz    # ADC map (will be calculated if not provided)

## File Naming Convention:
- The folder name should match the SubjectID
- All files should be prefixed with the SubjectID
- Files should be in NIfTI format (.nii or .nii.gz)

## Usage:
python codes/ADSRun.py -input "data/examples/Subject01/"

Note: The placeholder files in this directory are for demonstration only.
Replace them with actual medical imaging data for processing.
"""
        
        readme_path = example_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"   📂 Created example directory: {example_dir}")
        print(f"   📄 Created README.md with usage instructions")
        
        return True
    
    def generate_setup_report(self):
        """Generate setup validation report"""
        print("\n📊 Generating setup report...")
        
        # Run all checks
        checks = {
            "Python Version": self.check_python_version(),
            "Directory Structure": self.check_directory_structure(),
            "Required Files": self.check_required_files(),
            "Python Packages": self.requirements_installed,
            "TensorFlow GPU": self.check_tensorflow_gpu(),
            "Trained Models": self.check_trained_models(),
            "Template Files": self.check_template_files(),
            "Basic Functionality": self.test_basic_functionality()
        }
        
        # Create report
        report_content = f"""ADS Setup Report
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Python Version: {self.python_version}
Platform: {platform.system()} {platform.release()}

Setup Status:
{'='*50}
"""
        
        all_passed = True
        for check_name, status in checks.items():
            status_symbol = "✅" if status else "❌"
            report_content += f"{status_symbol} {check_name}: {'PASS' if status else 'FAIL'}\n"
            if not status:
                all_passed = False
        
        report_content += f"\nOverall Status: {'✅ READY' if all_passed else '❌ SETUP INCOMPLETE'}\n"
        
        if not all_passed:
            report_content += "\nNext Steps:\n"
            if not checks["Python Packages"]:
                report_content += "1. Install Python packages: python setup_ads.py --install-packages\n"
            if not checks["Trained Models"]:
                report_content += "2. Download trained models from: https://www.nitrc.org/projects/ads\n"
            if not checks["Template Files"]:
                report_content += "3. Download template files from the original repository\n"
        else:
            report_content += "\n🎉 Your ADS environment is ready!\n"
            report_content += "\nQuick Start:\n"
            report_content += "1. Place your data in: data/examples/SubjectID/\n"
            report_content += "2. Run: python codes/ADSRun.py -input data/examples/SubjectID/\n"
        
        # Save report
        report_path = self.project_root / "setup_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"   📄 Setup report saved to: {report_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 SETUP SUMMARY")
        print(f"{'='*60}")
        
        for check_name, status in checks.items():
            status_symbol = "✅" if status else "❌"
            print(f"{status_symbol} {check_name}")
        
        if all_passed:
            print(f"\n🎉 SUCCESS! Your ADS environment is ready to use!")
        else:
            print(f"\n⚠️  Setup incomplete. Please address the failed checks above.")
        
        return all_passed, checks
    
    def run_full_setup(self, install_packages=False, force_reinstall=False):
        """Run complete setup process"""
        print("🚀 Starting ADS Environment Setup...")
        print(f"Project directory: {self.project_root}")
        
        # Step 1: Basic checks
        if not self.check_python_version():
            return False
        
        # Step 2: Directory structure
        self.check_directory_structure()
        
        # Step 3: Required files
        if not self.check_required_files():
            print("\n❌ Cannot proceed without required Python files!")
            return False
        
        # Step 4: Install packages if requested
        if install_packages:
            self.install_requirements(force=force_reinstall)
        
        # Step 5: Create example structure
        self.create_example_data()
        
        # Step 6: Generate comprehensive report
        success, checks = self.generate_setup_report()
        
        return success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='setup_ads',
        description='Setup and validate ADS environment',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--install-packages', action='store_true',
                        help='Install Python packages from requirements.txt')
    
    parser.add_argument('--force-reinstall', action='store_true',
                        help='Force reinstall of all packages')
    
    parser.add_argument('--check-only', action='store_true',
                        help='Only check environment, do not install anything')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = ADSSetup()
    
    try:
        if args.check_only:
            # Only run checks
            success, checks = setup.generate_setup_report()
        else:
            # Run full setup
            success = setup.run_full_setup(
                install_packages=args.install_packages,
                force_reinstall=args.force_reinstall
            )
        
        if success:
            print(f"\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Setup completed with issues. See report for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()