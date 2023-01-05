import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numba import jit

import flowiz


def get_cli_arg():

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Data root directory relative path", type=str)
    parser.add_argument("--sat_dir", help="Satellite images directory name", type=str)
    parser.add_argument("--drone_dir", help="Drone images directory name", type=str)
    parser.add_argument("--label_dir", help="Satellite labels directory name", type=str)
    parser.add_argument("--output_dir", help="Output directory name", type=str)

    parser.add_argument("--img_type", help="sat or drone depending on the image type used", type=str)
    parser.add_argument("--ref_type", help="sat or drone depending on the image type used", type=str)

    parser.add_argument("--use_my_init", help="Whether to use custom initialization or default random one",
                        action='store_true')
    parser.add_argument("--use_my_rs", help="Whether to use custom search or default random search",
                        action='store_true')
    parser.add_argument("--debug", help="Use debug or not", action='store_true')

    parser.add_argument("--new_w", help="Output image width", type=int, default=1056)
    parser.add_argument("--new_h", help="Output image height", type=int, default=792)
    parser.add_argument("--itr", help="Number of oterations", type=int, default=5)
    parser.add_argument("--p_size", help="Window size for path match", type=int, default=3)
    parser.add_argument("--img_names", help="Predefined list of images, if empty, all images from drone dir are taken",
                        default=[], nargs="*")

    return parser.parse_args()


@jit(nopython=True)
def cal_distance(a, b, A_padding, B, p_size):
    """
    Distance function.
    """
    p = p_size // 2
    # Get patch around image index (start at upper left)
    patch_a = A_padding[a[0]:a[0] + p_size, a[1]:a[1] + p_size, :]
    # Get a patch around random reference index (start at center, 8 - neighbours)
    patch_b = B[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, :]
    # Calculate distance between image patch and random patch from reference, without taking nan's into consideration
    temp = patch_b - patch_a
    num = 0
    dist = 0
    for i in range(p_size):
        for j in range(p_size):
            for k in range(3):
                if not np.isnan(temp[i, j, k]):
                    num += 1
                    dist += np.square(temp[i, j, k])
    dist /= num
    return dist


def reconstruction(f, A, B, save_name='', output=False):
    """
    Funtion for reconstructing a new image from B (reference image) and f (correspondence map).
    """
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    temp = np.zeros_like(A)
    # Reconstruct the image based on indexes pairs at each i,j coordinate in correspondence map (f)
    for i in range(A_h):
        for j in range(A_w):
            temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
    if save_name != '':
        if output:
            output_cat = np.zeros((temp.shape[0], temp.shape[1]))
            output_cat[np.where(temp[:, :, 0] == 255)] = 1
            output_cat[np.where(temp[:, :, 1] == 255)] = 2
            output_cat[np.where(temp[:, :, 2] == 255)] = 3
            seg_img = Image.fromarray(output_cat.astype(np.uint8)).convert('P')
            seg_img.putpalette(np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8))
            seg_img.save('{}.png'.format(save_name))
        else:
            Image.fromarray(temp).save('{}.png'.format(save_name))


def initialization(A, B, p_size, use_own_init):
    """
    Initialization of the algorithm.
    """
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    # random.randint(low, high=None, size=None, dtype=int)
    # Create random maps with (max height - p) - B_r - and (max width - p) - B_c - indexes
    random_B_r = np.random.randint(p, B_h - p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w - p, [A_h, A_w])

    # Pad image with nan's
    A_padding = np.ones([A_h + p * 2, A_w + p * 2, 3]) * np.nan
    A_padding[p:A_h + p, p:A_w + p, :] = A

    # Initialize correspondence and distance maps with zeros
    f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])
    for i in range(A_h):
        for j in range(A_w):
            # Get an index from image
            a = np.array([i, j])
            # Get an index from the random maps (from reference), corresponding to index from image
            if use_own_init:
                if i == 0:
                    b_i = 1
                else:
                    b_i = i
                if j == 0:
                    b_j = 1
                else:
                    b_j = j
                if i == B_h or i == B_h - 1:
                    b_i = B_h - 2
                elif i != 0:
                    b_i = i
                if j == B_w or j == B_w - 1:
                    b_j = B_w - 2
                elif j != 0:
                    b_j = j
                b = np.array([b_i, b_j], dtype=np.int32)
            else:
                b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)

            # Fill correspondence map with the random index found above and compute distance
            f[i, j] = b
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding


@jit(nopython=True)
def inter(b_x, search_h, p, B_h, b_y, search_w, B_w, alpha, i, buildings_r_c, roads_r_c, class_fraction=8):
    """
    Funtion for creating the search space.
    """
    # Create searching patch of +- search_h and +- search_w around b_x, b_y reference pixel coordinates
    # and pick a random or custom index (x,y coordinates) from this patch
    search_min_r = max(b_x - search_h, p)
    search_max_r = min(b_x + search_h, B_h - p)
    search_min_c = max(b_y - search_w, p)
    search_max_c = min(b_y + search_w, B_w - p)
    patch_area = int((search_max_c - search_min_c) * (search_max_r - search_min_r))

    random_b_x = np.random.randint(search_min_r, search_max_r)
    random_b_y = np.random.randint(search_min_c, search_max_c)

    if buildings_r_c is not None and roads_r_c is not None:
        if b_x in buildings_r_c[0, :] and b_y in buildings_r_c[1, :]:
            # Check if buildings in patch
            building_in_patch = [buildings_r_c[:, i] for i in range(buildings_r_c.shape[1]) if (
                    math.ceil(search_min_r) <= buildings_r_c[0, i] < math.floor(search_max_r) and math.ceil(
                search_min_c) <= buildings_r_c[1, i] < math.floor(search_max_c))]

            if len(building_in_patch) > patch_area // class_fraction:
                building_in_patch = [point for point in building_in_patch if
                                     math.sqrt((b_x - point[0]) ** 2 + (b_y - point[1]) ** 2) < 15]
                if len(building_in_patch) > 0:
                    random_b_x, random_b_y = building_in_patch[np.random.choice(len(building_in_patch))]

        elif b_x in roads_r_c[0, :] and b_y in roads_r_c[1, :]:
            # Check if roads in patch
            road_in_patch = [roads_r_c[:, i] for i in range(roads_r_c.shape[1]) if (
                    math.ceil(search_min_r) <= roads_r_c[0, i] < math.floor(search_max_r) and math.ceil(
                search_min_c) <= roads_r_c[1, i] < math.floor(search_max_c))]

            if len(road_in_patch) > patch_area // class_fraction:
                road_in_patch = [point for point in road_in_patch if
                                 math.sqrt((b_x - point[0]) ** 2 + (b_y - point[1]) ** 2) < 15]
                if len(road_in_patch) > 0:
                    random_b_x, random_b_y = road_in_patch[np.random.choice(len(road_in_patch))]

    # Reduce the searching space at each iteration
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b = np.array([random_b_x, random_b_y])
    return search_h, search_w, b


def random_search(f, a, dist, A_padding, B, p_size, use_my_rs, alpha=0.5):
    """
    Function for the search step of the algorithm.
    """
    x = a[0]
    y = a[1]
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    # Take the appropriate pixel coordinates in the reference image
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    alpha = alpha
    while search_h > 1 and search_w > 1:
        building_arr, road_arr = None, None
        if use_my_rs:
            building_arr = np.vstack([use_my_rs[0][0], use_my_rs[0][1]])
            road_arr = np.vstack([use_my_rs[1][0], use_my_rs[1][1]])
        search_h, search_w, b = inter(b_x, search_h, p, B_h, b_y, search_w, B_w, alpha, i, building_arr, road_arr)
        # Compute distance between current patch and newly selected patch
        d = cal_distance(a, b, A_padding, B, p_size)
        # Update dist and correspondence map if new distance is less than current
        if d < dist[x, y]:
            dist[x, y] = d
            f[x, y] = b
        i += 1


def propagation(f, a, dist, A_padding, B, p_size, is_odd):
    """
    Function for the propagation step of the algorithm.
    """
    p = p_size // 2
    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    x = a[0]
    y = a[1]
    # If iteration is odd number (iterations start with 1)
    if is_odd:
        # Take upper cell from current one in f if possible
        i_left, j_left = f[max(x - 1, 0), y]
        # Increase row index by one, take the next below
        i_left = min(i_left + 1, A_h - 1 - p)
        b_left = np.array([i_left, j_left], dtype="int64")
        # Compute distance
        d_left = cal_distance(a, b_left, A_padding, B, p_size)

        # Take left cell from current one in f if possible
        i_up, j_up = f[x, max(y - 1, 0)]
        # Increase column index by one, take the next to the right
        j_up = min(j_up + 1, A_w - 1 - p)
        b_up = np.array([i_up, j_up], dtype="int64")
        # Compute distance
        d_up = cal_distance(a, b_up, A_padding, B, p_size)

        # Take the minimum between all 3 distances
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        # Update f and dist according to minmum distance found
        if idx == 1:
            f[x, y] = b_left
            dist[x, y] = d_left
        if idx == 2:
            f[x, y] = b_up
            dist[x, y] = d_up
    # If iteration is even number (iterations start with 1)
    else:
        # Take lower cell from current one in f if possible
        i_right, j_right = f[min(x + 1, A_h - 1), y]
        # Decrease row index by one, take the next below
        i_right = max(i_right - 1, p)
        b_right = np.array([i_right, j_right], dtype="int64")
        # Compute distance
        d_right = cal_distance(a, b_right, A_padding, B, p_size)

        # Take right cell from current one in f if possible
        i_down, j_down = f[x, min(y + 1, A_w - 1)]
        # Decrease column index by one, take the next to the left
        j_down = max(j_down - 1, p)
        b_down = np.array([i_down, j_down], dtype="int64")
        # Compute distance
        d_down = cal_distance(a, b_down, A_padding, B, p_size)

        # Take the minimum between all 3 distances
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y] = b_right
            dist[x, y] = d_right
        if idx == 2:
            f[x, y] = b_down
            dist[x, y] = d_down


def debug_fct(f, img, ref, save_name, debug_step, read_label):
    """
    Function used for debugging.
    """
    if read_label is not None:
        reconstruction(f, img, read_label, 'debug/{}_{}'.format(save_name, debug_step))
    else:
        reconstruction(f, img, ref, 'debug/{}_{}'.format(save_name, debug_step))


def NNS(img, ref, p_size, itr, use_own_f_init, save_name, use_my_rs_idx, img_label, debug):
    """
    Function with the whole Patch Match pipeline.
    :param img: Image
    :param ref: Reference image
    :param p_size: Window size
    :param itr: Number of iterations of the algorithm
    :param use_own_f_init: Whether to use an initialisation equal to the reference image, if True,
    or a random one if False
    :param save_name: Name used when saving the warped image
    :param use_my_rs_idx: Buildings and roads indexes if using a constrained search, or None
    :param img_label: Whether to read the reference (label) image or not, for debugging purposes
    :param debug: True if debugging is wanted
    :return:
    """
    A_h = np.size(img, 0)
    A_w = np.size(img, 1)
    if debug:
        print("initialization")
    f, dist, img_padding = initialization(img, ref, p_size, use_own_f_init)
    if debug:
        print(f"score : {dist.mean()}")
    score_list = []
    read_label = None

    if 'drone_sat' in save_name:
        read_label = img_label

    if debug:
        debug_fct(f, img, ref, save_name, 'after_init', read_label)

    for itr in range(1, itr + 1):
        if debug:
            print("iteration: %d" % (itr))
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    a = np.array([i, j])
                    # Change f and dist according to the iteration
                    propagation(f, a, dist, img_padding, ref, p_size, False)
                    random_search(f, a, dist, img_padding, ref, p_size, use_my_rs_idx)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    a = np.array([i, j])
                    # Change f and dist according to the iteration
                    propagation(f, a, dist, img_padding, ref, p_size, True)
                    random_search(f, a, dist, img_padding, ref, p_size, use_my_rs_idx)

                if debug and (i == A_h // 2 or i == A_h // 4 or i == (3 * A_h // 4)) and itr == 1:
                    debug_fct(f, img, ref, save_name, 'after_iter{}_{}'.format(itr, i), read_label)

        if debug:
            print(f"score : {dist.mean()}")
        score_list.append(dist.mean())

        if debug:
            debug_fct(f, img, ref, save_name, 'after_iter{}_end'.format(itr), read_label)

    return f, dist, score_list


def alter_flow(flow):
    """
    Function for altering the obtained flow for a suitable input for the flowviz library.
    """
    h, w = flow.shape
    raw_indexes_h = np.pad(np.tile(np.expand_dims(np.arange(1, h - 1), axis=1), w), ((1, 1), (0, 0)),
                           'constant', constant_values=(1, h - 2))
    raw_indexes_w = np.pad(np.repeat(np.expand_dims(np.arange(1, w - 1), axis=0), h, axis=0), ((0, 0), (1, 1)),
                           'constant', constant_values=(1, w - 2))
    u = np.zeros((h, w))
    v = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            u[i, j], v[i, j] = flow[i, j]
    u = u - raw_indexes_h
    v = v - raw_indexes_w

    return np.stack([u, v], axis=-1)


def get_classes_idx(im):
    """
    Function that returns buildings and roads indexes in image
    """
    # Threshold and get class idx
    r = np.where(im == 2, 255, 0)
    b = np.where(im == 1, 255, 0)
    class_im = np.stack([r, np.zeros((im.shape[0], im.shape[1])), b]).transpose((1, 2, 0)).astype(np.uint8)
    buildings_idx = np.where(class_im[:, :, 2])
    roads_idx = np.where(class_im[:, :, 0])
    return [buildings_idx, roads_idx]


def threshold_img(img_hsv):
    """
    Function for obtaining categorical label based on HSV channel thresholding
    """
    # define range of blue color in HSV
    lower_blue = np.array([115, 50, 50])
    upper_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # define range of green color in HSV
    lower_green = np.array([55, 50, 50])
    upper_green = np.array([65, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([5, 255, 255])
    mask0_red = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([175, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1_red = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask_red = mask0_red + mask1_red

    # set my output img to zero everywhere except my mask
    output_hsv = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))
    output_hsv[np.where(mask_red == 255)] = 1
    output_hsv[np.where(mask_blue == 255)] = 2
    output_hsv[np.where(mask_green == 255)] = 3

    return output_hsv.astype(np.uint8)


def main(data_dir, sat_dir, drone_dir, label_dir, output_dir, img_type,
         ref_type, use_my_rs, use_my_init, debug, itr, p_size, new_w, new_h, img_names=[]):
    """
    Main function for generating the warped labels.
    """

    debug_folder = 'debug_viz'
    sat_path = os.path.join(data_dir, sat_dir)
    drone_path = os.path.join(data_dir, drone_dir)
    label_path = os.path.join(data_dir, label_dir)
    os.makedirs(debug_folder, mode=0o777, exist_ok=True)
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    # Set files to compute the correspondence
    if len(img_names) != 0:
        files = img_names
    else:
        files = os.listdir(drone_path)

    for im in files:
        img_name = im.split('.')[0]

        if img_type == 'sat':
            img_path = os.path.join(sat_path, im)
        else:
            img_path = os.path.join(drone_path, im)
        if ref_type == 'sat':
            ref_path = os.path.join(sat_path, im)
        else:
            ref_path = os.path.join(drone_path, im)

        # Read images and resize
        img = np.array(Image.open(img_path).resize((new_w, new_h)))
        ref = np.array(Image.open(ref_path).resize((new_w, new_h)))
        img_label = Image.open(os.path.join(label_path, img_name + '.png')).resize((new_w, new_h))

        if use_my_rs:
            use_my_rs_idx = get_classes_idx(np.array(img_label))
        else:
            use_my_rs_idx = None

        img_label = np.array(img_label.convert('RGB'))

        # Build the save name based on configuration
        save_name = '{}-{}_{}_psize{}_{}iter{}{}'.format(img_name, img_type, ref_type, p_size, itr,
                                                         '_myinit' if use_my_init else '',
                                                         '_myrs' if use_my_rs_idx is not None else '')

        # Find correspondence map and distance map
        f, dist, score_list = NNS(img, ref, p_size, itr, use_my_init, save_name, use_my_rs_idx, img_label, debug)
        # Reconstruct drone labels
        reconstruction(f, img, img_label, os.path.join(output_dir, save_name.split('-')[0]),
                       output='drone_sat' in save_name)
        print('{} done!'.format(im))

        if debug:
            reconstruction(f, img, ref, os.path.join(debug_folder, 'corrected_{}'.format(save_name)))

            # Visualize distance map
            plt.imshow(dist / 255)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_folder, 'dist_{}.png'.format(save_name)))

            # Visualize flow
            flow = alter_flow(f)
            flow_img = flowiz.convert_from_flow(flow)
            # Flow viz description (based on displacement between rows and column from drone and satellite images)
            # row high, col high - red/yellow
            # row lower, col higher - green/cyan
            # row higher, col lower - pink/violet
            # row low, col low - blue/dark blue
            Image.fromarray(flow_img).save(os.path.join(debug_folder, 'flow_{}.jpg'.format(save_name)))
            plt.close()

            # Plot average distance score
            plt.plot(score_list)
            plt.tight_layout()
            plt.savefig(os.path.join(debug_folder, 'scores_{}.png'.format(save_name)))
        break

if __name__ == "__main__":
    args = get_cli_arg()

    main(args.data_dir, args.sat_dir, args.drone_dir, args.label_dir, args.output_dir, args.img_type, args.ref_type,
         args.use_my_rs, args.use_my_init, args.debug, args.itr, args.p_size, args.new_w, args.new_h,
         img_names=args.img_names)
