import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define paths to your training data
REAL_DIR = "dataset/spectrograms/train/real"
FAKE_DIR = "dataset/spectrograms/train/fake"

def inspect_images(num_samples=5):
    if not os.path.exists(REAL_DIR) or not os.path.exists(FAKE_DIR):
        print(f"Error: Could not find directories.\nChecked: {REAL_DIR}\nChecked: {FAKE_DIR}")
        return

    real_files = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith('.png')]
    fake_files = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith('.png')]

    if not real_files or not fake_files:
        print("Error: One of the folders is empty.")
        return

    # Select random samples
    real_samples = random.sample(real_files, min(len(real_files), num_samples))
    fake_samples = random.sample(fake_files, min(len(fake_files), num_samples))

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle("Top Row: Real | Bottom Row: Fake", fontsize=16)

    for i in range(num_samples):
        # Real
        axes[0, i].imshow(mpimg.imread(real_samples[i]))
        axes[0, i].axis('off')
        # Fake
        axes[1, i].imshow(mpimg.imread(fake_samples[i]))
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inspect_images()