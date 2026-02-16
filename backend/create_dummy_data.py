import os
import numpy as np
from PIL import Image
import soundfile as sf

def create_dummy_images(base_path, num_samples=10):
    """Creates dummy spiral/wave images for testing."""
    categories = ['spiral', 'wave']
    labels = ['healthy', 'parkinson']
    
    for category in categories:
        for label in labels:
            path = os.path.join(base_path, 'drawings', category, label)
            os.makedirs(path, exist_ok=True)
            for i in range(num_samples):
                # Create a random image
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(path, f'dummy_{i}.png'))
    print(f"Created {num_samples} dummy images per class in {base_path}/drawings")

def create_dummy_audio(base_path, num_samples=10):
    """Creates dummy audio files for testing."""
    labels = ['healthy', 'parkinson']
    sr = 22050
    duration = 3 # seconds
    
    for label in labels:
        path = os.path.join(base_path, 'voice', label)
        os.makedirs(path, exist_ok=True)
        for i in range(num_samples):
            # Create random noise as audio
            audio = np.random.uniform(-1, 1, int(sr * duration))
            sf.write(os.path.join(path, f'dummy_{i}.wav'), audio, sr)
    print(f"Created {num_samples} dummy audio files per class in {base_path}/voice")

if __name__ == "__main__":
    base_dir = "datasets"
    create_dummy_images(base_dir)
    create_dummy_audio(base_dir)
