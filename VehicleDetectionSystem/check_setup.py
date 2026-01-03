
import os
import sys
from pathlib import Path

def check_requirements():
    print(" Checking requirements...\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if model exists
    model_path = Path("models/best.pt")
    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"✗ Model NOT found at {model_path}")
        print("  Please copy your model to models/best.pt")
        return False
    
    # Check required packages
    required_packages = [
        'streamlit',
        'ultralytics',
        'cv2',
        'PIL',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ Package '{package}' installed")
        except ImportError:
            print(f"✗ Package '{package}' NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check directories
    print("\n Checking directories...")
    dirs = ['models', 'results', 'uploads', 'utils']
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"✗ Directory '{dir_name}' missing")
            os.makedirs(dir_name, exist_ok=True)
            print(f"  Created '{dir_name}'")
    
    print("\n All requirements met!")
    print("\n" + "="*50)
    print(" Ready to launch! Run:")
    print("   streamlit run app.py")
    print("="*50)
    return True

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    check_requirements()
