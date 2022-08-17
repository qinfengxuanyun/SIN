import os
import os.path
import csv
import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files,phase="test"):

    if phase == "train":
        if not os.path.exists(path_files+'train.txt'):
            paths, size = make_dataset_dir(path_files)
        else:
            paths, size = make_dataset_txt(path_files + 'train.txt')
    if phase == "val":
        if not os.path.exists(path_files+'val.txt'):
            paths, size = make_dataset_dir(path_files)
        else:
            paths, size = make_dataset_txt(path_files + 'val.txt')
    if phase == "test":
        #paths, size = make_dataset_dir(path_files)
        if not os.path.exists(path_files+'test.txt'):
            paths, size = make_dataset_dir(path_files)
        else:
            paths, size = make_dataset_txt(path_files + 'test.txt')

    #paths, size = make_dataset_dir(path_files)
    return paths, size


def make_dataset2(path_files):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size

def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths, len(img_paths)

def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        fnames = sorted(fnames)
        for index in range(0, len(fnames)):
            fname = fnames[index]
            if is_image_file(fname):
                path = fname#os.path.join(root, fname)
                img_paths.append(path)
    return img_paths, len(img_paths)

