import cv2
import os


def overlay_images(images_path, img_list, name_list):
    """
    Function for overlaying muliple pairs of images.
    :param images_path: Path to directory of the images to be overlayed.
    :param img_list: List of pairs of tuples, where first element in tuple is the image(.jpg)
     and the second one the label(.png)
    :param name_list: List of strings of name suffixes for each overlay created from the img_list
    :return:
    """
    assert len(img_list) == len(name_list), "The 2 lists must be equal"

    for i, img_tuple in enumerate(img_list):
        img1_name, img2_name = img_tuple[0], img_tuple[1]
        img1 = cv2.imread(os.path.join(images_path, img1_name))
        img2 = cv2.imread(os.path.join(images_path, img2_name))
        dst = cv2.addWeighted(img1, 0.8, img2, 0.6, 0)
        cv2.imwrite(os.path.join(images_path, name_list[i] + '_' + img1_name), dst)


if __name__ == '__main__':
    overlay_images('../drone_dataset/visualization', [('DJI_0337_120.jpg', 'DJI_0337_120_h.png')], ['overlay_h'])
