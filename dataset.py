import matplotlib.pyplot as plt
import torchvision.datasets

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def show_image(dataset, num_samples=20, cols=4):
    """Plot random samples from the dataset"""
    # Create a image window with a size of 15x15 inches, which will be used to
    # display the subsequently drawn image content
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        # Create a subplot grid in the figure to display
        # multiple images in a structured layout.
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])
    plt.show()


data = torchvision.datasets.StanfordCars(root=".", download=False)
show_image(data)