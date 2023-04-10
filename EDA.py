from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
Exploratory Data Analysis

Background: 
The MNIST dataset is a set of 70,000 images of digits handwritten by high school students
and employees of the US Census Bureau, and each has 784 features.
Because each image is 28 x 28 pixel and each feature represents one pixel's intensity
from 0(white) to 255(black)

Python's open-source library Scikit-Learn provides many helper functions to download popular datasets.
MNIST is one of them. 
We will be using the built-in dataset for this project.

data.keys() function contains an array with one row per instance and one column per feature,
here is the one for MNIST:
dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

Shape of X(data) dataset:  (70000, 784)
Shape of y(target) dataset:  (70000,)

"""

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X, y = mnist["data"], mnist["target"]
# cast y to integers
y = y.astype(np.uint8)

# Take a look of the dataset
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images)
plt.show()