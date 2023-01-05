import os
import shutil
import argparse
import numpy as np
from PIL import Image


def get_cli_arg():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_dataset_root", help="Dataset root directory (current)", type=str)
    parser.add_argument("--old_images_root", help="Drone images, satellite images, maperitive labels,"
                        " raster images directory (current)", type=str)
    parser.add_argument("--new_dataset_name", help="Dataset name in the new folder structure", type=str)
    parser.add_argument("--data_root", help=" Dataset root directory", type=str, default='data')
    parser.add_argument("--img_dir_name", help="Satellite images directory", type=str, default='img_dir')
    parser.add_argument("--ann_dir_name", help="Satellite images directory", type=str, default='ann_dir')
    parser.add_argument("--drone_dir_name", help="Drone images directory", type=str, default='drone_dir')

    return parser.parse_args()


def rename_duplicates_morges_vevey(dataset_root):
    """
    Some images in 'morges' directory have the same name as in 'vevey'. This function renames those files.
    :param dataset_root: Dataset root directory
    """

    for city_dir_name in ['vevey', 'morges']:
        data_path = os.path.join(dataset_root, city_dir_name)
        files = sorted(os.listdir(data_path))
        # We defined all the common image names to vevey and morges
        vevey_morges_overlap = ['DJI_0{}_120'.format(i) for i in range(256, 265)]
        vevey_morges_overlap.extend(['DJI_0{}_120'.format(i) for i in range(267, 276)])
        vevey_morges_overlap.extend(['DJI_0{}_80'.format(i) for i in range(284, 305)])

        for file in files:
            filename = file.split('.')[0]
            extension = file.split('.')[1]
            if filename in vevey_morges_overlap:
                # Create new name according to the folder the image is coming from
                new_name = '_'.join(filename.split('_')[:3]) + '_z_' + city_dir_name + '.' + extension
                os.rename(os.path.join(data_path, file), os.path.join(data_path, new_name))


def change_human_drone_anns(dataset_root, human_label_dir, data_root='data', new_human_dir='human_drone_dir'):
    """
    Change the format of human annotated drone labels and save them.
    :param dataset_root: Dataset root directory (current)
    :param human_label_dir: Human annotations for drone images directory name
    :param data_root: New data root directory
    :param new_human_dir: New name for human labels directory
    """
    labels_dir = os.path.join(dataset_root, human_label_dir)
    new_human_labels_dir = os.path.join('../', data_root, new_human_dir)
    for file in os.listdir(labels_dir):
        filename = file.split('.')[0]
        new_file = '_'.join(filename.split("_")[:3]) + '.' + file.split('.')[1]
        # Read image
        label = Image.open(os.path.join(labels_dir, file))
        label_array = np.array(label)
        # Transform into categorical format
        output_cat = np.zeros((label_array.shape[0], label_array.shape[1]))
        output_cat[np.where(label_array[:, :, 0] == 255)] = 1
        output_cat[np.where(label_array[:, :, 1] == 255)] = 2
        output_cat[np.where(label_array[:, :, 2] == 255)] = 3
        # Convert to 'P' mode and add the palette
        seg_img = Image.fromarray(output_cat.astype(np.uint8)).convert('P')
        seg_img.putpalette(np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8))
        seg_img.save(os.path.join(new_human_labels_dir, new_file))


def organise_dir_struct(old_dataset_root, old_images_root, new_dataset_name, data_root='data', img_dir_name='img_dir',
                        ann_dir_name='ann_dir', drone_dir_name='drone_dir'):
    """
    Function to organise the (current) data into the specific format for the warping or semantic segmentation repos.
    :param old_dataset_root: Dataset root directory (current)
    :param old_images_root: Drone images, satellite images, maperitive labels, raster images directory (current)
    :param new_dataset_name: Dataset name in the new folder structure
    :param data_root: New data root directory
    :param img_dir_name: Satellite images directory
    :param ann_dir_name: Satellite labels directory
    :param drone_dir_name: Drone images directory
    """

    # Declare directory paths and create directories if necessary
    all_img_dir = os.path.join(old_dataset_root, old_images_root)

    data_dir = os.path.join('../', data_root)
    dataset_dir = os.path.join(data_dir, new_dataset_name)
    img_dir = os.path.join(dataset_dir, img_dir_name)
    ann_dir = os.path.join(dataset_dir, ann_dir_name)
    drone_dir = os.path.join(dataset_dir, drone_dir_name)

    os.makedirs(data_dir, mode=0o777, exist_ok=True)
    os.makedirs(dataset_dir, mode=0o777, exist_ok=True)
    os.makedirs(img_dir, mode=0o777, exist_ok=True)
    os.makedirs(ann_dir, mode=0o777, exist_ok=True)
    os.makedirs(drone_dir, mode=0o777, exist_ok=True)

    for filename in os.listdir(all_img_dir):
        # Rename (if necessary) each file and move to the specific location
        if 'DNG' in filename:
            pass
        elif 'satellite' in filename:
            img_name = "_".join(filename.split('.')[0].split('_')[:-1]) + '.jpg'
            if not os.path.exists(os.path.join(img_dir, img_name)):
                shutil.copy(os.path.join(all_img_dir, filename), os.path.join(img_dir, img_name))
        elif 'maperitive' in filename:
            img_name = "_".join(filename.split('.')[0].split('_')[:-1]) + '.png'
            if not os.path.exists(os.path.join(ann_dir, img_name)):
                shutil.copy(os.path.join(all_img_dir, filename), os.path.join(ann_dir, img_name))
        else:
            img_name = filename.split('.')[0] + '.jpg'
            if not os.path.exists(os.path.join(drone_dir, img_name)):
                shutil.copy(os.path.join(all_img_dir, filename), os.path.join(drone_dir, img_name))
        # Delete the files in the old location
        os.remove(os.path.join(all_img_dir, filename))
    os.rmdir(all_img_dir)


if __name__ == '__main__':
    args = get_cli_arg()
    organise_dir_struct(args.old_dataset_root, args.old_images_root, args.new_dataset_name, args.data_root,
                        args.img_dir_name, args.ann_dir_name, args.drone_dir_name)
