import os
import tqdm
import dji_utils
from label import get_bboxes
import argparse


def get_cli_arg():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", help="Dataset directory relative path", type=str)
    parser.add_argument("--images_dir", help="Images, labels, rasters directory name", type=str)
    parser.add_argument("--splits_dir", help="Splits directory name", type=str)

    return parser.parse_args()


class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2, y2)

    __and__ = intersection

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))


def get_filenames(data_root, train=True, val=True, splits_dir='splits'):
    """
    Function for reading all train and/or val samples from the txt files
    :param data_root: Dataset root directory
    :param train: True if train ssplit should be read
    :param val: True if val ssplit should be read
    :param splits_dir: Split txt files directory
    :return: List of samples wanted, according to split
    """
    splits_dir = os.path.join(data_root, splits_dir)
    data_val = []
    data_train = []

    if val:
        with open(os.path.join(splits_dir, 'val.txt'), 'r') as f:
            data_val = f.readlines()
            data_val = [line[:-1] for line in data_val]
    if train:
        with open(os.path.join(splits_dir, 'train.txt'), 'r') as f:
            data_train = f.readlines()
            data_train = [line[:-1] for line in data_train]

    return sorted(data_train + data_val)


def train_test_overlap(data_path, img_dir, train_imgs=None, val_imgs=None):
    """
    Function to compute the overlap between validation and train images.
    :param data_path: Dataset root directory
    :param img_dir: Drone images, satellite images, maperitive labels, raster images directory
    :param train_imgs: List of train images names. If None, all images names from img_dir
     are taken into consideration (excluding the ones used for validation)
    :param val_imgs: List of val images names. If None, images already defined in a text file
     val.txt will be taken into consideration.
    :return: Lists of non-overlapping train images and val images
    """
    train_val_path = os.path.join(data_path, img_dir)
    if val_imgs is None:
        with open('val.txt', 'r') as f:
            data_val = f.readlines()
            val_imgs = sorted([line[:-1] for line in data_val])

    if train_imgs is None:
        all_imgs = os.listdir(train_val_path)
        train_imgs = set([img.split('.')[0] for img in all_imgs
                          if img.split('.')[0] not in val_imgs and img.split('.')[1] == 'JPG'])
        print(len(train_imgs))
    overlap = False
    overlap_val_train = {}

    # Iterate over all validation images and check if there is overlap with each train image
    for i, val_im in tqdm.tqdm(enumerate(val_imgs)):
        dng = os.path.join(train_val_path, val_im + '.DNG')
        jpg = os.path.join(train_val_path, val_im + '.JPG')
        metadata = dji_utils.get_dji_metadata(dng)
        val_im = dji_utils.DJIImage(val_im, dng, jpg, metadata)
        # Get validation box coordinates
        extent_val, _, _, _, _ = get_bboxes(val_im)
        a_val = Rectangle(extent_val[2], extent_val[0], extent_val[3], extent_val[1])
        for j, train_im in enumerate(train_imgs):
            dng = os.path.join(train_val_path, train_im + '.DNG')
            jpg = os.path.join(train_val_path, train_im + '.JPG')
            metadata = dji_utils.get_dji_metadata(dng)
            train_im = dji_utils.DJIImage(train_im, dng, jpg, metadata)
            # Get train box coordinates
            extent_train, _, _, _, _ = get_bboxes(train_im)
            b_train = Rectangle(extent_train[2], extent_train[0], extent_train[3], extent_train[1])
            # Check if overlap exists and build a set of overlapping train images for each val image
            if a_val & b_train is not None:
                if val_im.img_name not in overlap_val_train:
                    overlap_val_train[val_im.img_name] = set()
                overlap_val_train[val_im.img_name].add(train_im.img_name)
                overlap = True
    if not overlap:
        print("No overlapping between train and test!")
        new_train = train_imgs
    else:
        print("Overlap val:train is \n {}".format(overlap_val_train))
        overlapped_train = set()
        # Create a set with all overlapping train images
        for val_im in overlap_val_train:
            overlapped_train.update(overlap_val_train[val_im])
        # Get a set of non-overlapping train images
        new_train = train_imgs.difference(overlapped_train)

    print("Train length is {}".format(len(new_train)))
    print("Val length is {}".format(len(val_imgs)))

    # 14 of old 120 are still overlapping ( 61/75 are ok with val )
    # 24 of all 120 are still overlapping ( 106/130 are ok with val ) but I keep 14 from before
    #

    return new_train, val_imgs


def clean_overlap(train_imgs, val_imgs, data_path, img_dir):
    """
    Function for deleting unnecessary images, from a directory where all of them are stored (img_dir).

    :param train_imgs: List of train images names (without extension)
    :param val_imgs: List of val images names (without extension)
    :param data_path: Dataset root directory
    :param img_dir: Drone images, satellite images, maperitive labels, raster imgaes directory
    :return:
    """
    all_imgs = train_imgs.union(val_imgs)
    train_val_path = os.path.join(data_path, img_dir)
    for filename in os.listdir(train_val_path):
        img_name = filename.split('.')[0]
        if img_name not in all_imgs:
            os.remove(os.path.join(train_val_path, filename))


def create_splits(train_imgs, val_imgs, data_path, split_dir='splits'):
    """
    Function for creating train/val splits txt files.
    :param train_imgs: List of train images names (without extension)
    :param val_imgs: List of validation images names (without extension)
    :param data_path: Dataset root directory
    :param split_dir: Name of directory containing the new train.txt and val.txt
    """

    os.makedirs(os.path.join(data_path, split_dir), mode=0o777, exist_ok=True)
    # Create train.txt split
    with open(os.path.join(data_path, split_dir, 'train.txt'), 'w') as f:
        f.writelines(im + '\n' for im in train_imgs)
    # Create val.txt split
    with open(os.path.join(data_path, split_dir, 'val.txt'), 'w') as f:
        f.writelines(im + '\n' for im in val_imgs)


if __name__ == "__main__":
    args = get_cli_arg()

    train_clean, val_clean = train_test_overlap(args.dataset_root, args.images_dir)
    clean_overlap(train_clean, val_clean, args.dataset_root, args.images_dir)
    create_splits(train_clean, val_clean, args.dataset_root, split_dir=args.splits_dir)
