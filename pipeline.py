import numpy as np
import os
from os import listdir
from os.path import join, isdir
from skimage.io import imread, imsave
from skimage.filters import threshold_yen
from skimage.morphology import closing, square
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.transform import resize


def get_files_labels(data_directory, list_class_names):
    """Searches data_dir for directories named after the entries of classes and returns the file paths and labels."""

    files = []
    labels = []
    for label_name in list_class_names:
        class_path = join(data_directory, label_name)

        # Create list of file paths
        file_list = [
            join(class_path, f)
            for f in listdir(join(class_path))
            if f.endswith('.jpeg')
        ]
        # Create label list with equal length
        label_list = [label_name] * len(file_list)

        files.extend(file_list)
        labels.extend(label_list)

    return files, labels


def skimage_cropping(filepath, target_height, target_width):
    # First crop
    img = imread(filepath)
    image_crop = img[160:160 + 880, 580:580 + 820]

    # Masking/closing pixel regions and labeling pixels
    thresh = threshold_yen(image_crop)
    img_closing = closing(image_crop > thresh, square(3))
    img_label = sk_label(img_closing)

    # Search for biggest area and extract centroid
    max_area = 0
    for region in regionprops(img_label):
        if region.area > max_area:
            max_area = region.area
            biggest = region
    center = biggest.centroid

    # Draw square bounding box around centroid
    square_side = 300
    step = square_side / 2
    min_row, max_row, min_col, max_col = max([0, int(center[0]-step)]),\
                                         int(center[0]+step), \
                                         max([0, int(center[1]-step)]),\
                                         int(center[1]+step)

    # Crop and resize image to square bounding box
    image_square = image_crop[min_row:max_row, min_col:max_col]
    image_resize = resize(image_square, [target_height, target_width], preserve_range=True).astype(np.uint8)

    return image_resize


target_height = 200
target_width = 200

data_dir = 'nailgun'
new_path = 'nailgun_processed_{}'.format(target_height)
classes = [
    'good',
    'bad'
]

good_dir = join(new_path, classes[0])
bad_dir = join(new_path, classes[1])

if not isdir(good_dir):
    os.makedirs(good_dir)
if not isdir(bad_dir):
    os.makedirs(bad_dir)

files, labels = get_files_labels(data_dir, classes)

for i, (file, label) in enumerate(zip(files, labels)):
    image = skimage_cropping(file, target_height, target_width)
    if label == 'good':
        file_path = join(good_dir, 'good_{}.jpeg'.format(i))
    elif label == 'bad':
        file_path = join(bad_dir, 'bad_{}.jpeg'.format(i))
    else:
        print('Else?')
        break
    imsave(file_path, image)

