import os
import sys
import subprocess
import requests
from pathlib import Path

class PoseEstimationSetup:
    def __init__(self):
        self.requirements = [
            'opencv-python',
            'numpy',
            'openvino',
            'ipython'
        ]
        self.base_model_dir = Path("model")
        self.model_name = "human-pose-estimation-0001"
        self.precision = "FP16-INT8"
        
    def install_requirements(self):
        """Install required Python packages"""
        print("Installing required packages...")
        for package in self.requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def download_notebook_utils(self):
        """Download notebook utilities"""
        print("Downloading notebook utilities...")
        utils_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
        r = requests.get(url=utils_url)
        with open("notebook_utils.py", "w") as f:
            f.write(r.text)

    def download_model(self):
        """Download pose estimation model"""
        print("Downloading pose estimation model...")
        model_path = self.base_model_dir / "intel" / self.model_name / self.precision / f"{self.model_name}.xml"
        
        if not model_path.exists():
            # Create directories if they don't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{self.model_name}/{self.precision}/"
            
            # Download model files
            for ext in ['.xml', '.bin']:
                url = model_url_dir + self.model_name + ext
                output_path = model_path.parent / (self.model_name + ext)
                
                print(f"Downloading {url}...")
                response = requests.get(url)
                with open(output_path, 'wb') as f:
                    f.write(response.content)

    def setup(self):
        """Run complete setup"""
        try:
            self.install_requirements()
            self.download_notebook_utils()
            self.download_model()
            print("Setup completed successfully!")
            return True
        except Exception as e:
            print(f"Setup failed with error: {str(e)}")
            return False

def main():
    setup = PoseEstimationSetup()
    setup.setup()

if __name__ == "__main__":
    main()