from __future__ import print_function

import argparse
import os
import posixpath
import shutil
import subprocess
from pathlib import Path
import numpy as np
import cartopy.io.img_tiles as cimgt
import dji_utils
import requests
from PIL import Image, ImageTransform
from cartopy_help import image_spoof, map_query
from map_alignment import get_extent, rot_and_crop
import matplotlib.pyplot as plt
import cv2


def get_cli_arg():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir_path", help="Images to process directory relative path",
                        type=str)
    parser.add_argument("--scale", help="Scale for Maperitive. "
                                        "Increase it the OSM temporary maps are too small, or "
                                        "decrease it if they are too big (default : 4).",
                        type=int, default=4)
    parser.add_argument("--rules", help="Relative path for rulefile for Maperitive.",
                        type=str, default="./custom.mrules")
    parser.add_argument("--maperitive", help="Maperitive binary relative path.",
                        type=str)
    parser.add_argument("--access_token", help="Access token for MapBox", type=str, default='')
    parser.add_argument('--raster', action='store_true')
    parser.add_argument('--label', action='store_true')
    parser.add_argument('--satellite', action='store_true')
    args = parser.parse_args()

    images_dir_path = os.path.abspath(args.images_dir_path)
    dest_labels_rasters = os.path.abspath(args.images_dir_path)

    scale = args.scale
    raster, label, satellite = args.raster, args.label, args.satellite
    maperitive_binary = os.path.abspath(args.maperitive)
    rulefile = os.path.abspath(args.rules)
    access_token = args.access_token

    return images_dir_path, dest_labels_rasters, scale, maperitive_binary,\
        rulefile, raster, label, satellite, access_token


def labeling_setup():
    print("")
    print("###############################################")
    print("##### Generating labels and raster images! #####")
    print("###############################################")
    print("")

    # Get CLI arguments
    images_dir_path, dest_labels_rasters, scale, maperitive_binary, rulefile,\
        raster, label, satellite, access_token = get_cli_arg()

    # Import images
    imgs = []

    # Convert to Unix file path style
    images_dir_path = images_dir_path.replace(os.sep, posixpath.sep)
    imgs.extend(dji_utils.get_imgs(images_dir_path))

    # Setup Cartopy
    cimgt.OSM.get_image = image_spoof  # reformat web request for street map spoofing
    osm_img = cimgt.OSM()  # spoofed, downloaded street map

    # Setup OverPass requests header
    useragent = 'EPFL/MihaiDavid'
    headers = {
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Google Chrome 80"',
        'Accept': '*/*',
        'Sec-Fetch-Dest': 'empty',
        'User-Agent': useragent,
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'https://overpass-turbo.eu',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'cors',
        'Referer': 'https://overpass-turbo.eu/',
        'Accept-Language': '',
        'dnt': '1',
    }

    return imgs, dest_labels_rasters, scale, maperitive_binary, rulefile, headers,\
        osm_img, raster, label, satellite, access_token


def get_bboxes(img):
    # Get bounding box coordinates based on GPS coordinates
    center_pt = [img.metadata['GpsLatitude'], img.metadata['GpsLongitude']]
    pic = Image.open(img.jpg_path)
    extent, outer_earth_meters, inner_earth_meters = get_extent(center_pt, img.metadata['FlightYawDegree'],
                                                                img.metadata['RelativeAltitude'])
    return extent, outer_earth_meters, inner_earth_meters, pic, center_pt


def call_maperitive(extent, source, rulefile, dest_labels_rasters, scale, maperitive_binary, filename, margin):
    # Prepare code for the Maperitive script subprocess call
    coords = [[extent[0], extent[2]], [extent[1], extent[3]]]
    ms = ['set-setting name=map.decoration.grid value=False', 'use-ruleset location={}'.format(rulefile),
          'load-source {}'.format(source), 'set-setting name=map.decoration.attribution value=false',
          'set-setting name=map.decoration.scale value=false']
    lngs, lats = zip(*coords)
    minlng, minlat = min(lngs), min(lats)
    maxlng, maxlat = max(lngs), max(lats)
    ms.append('set-geo-bounds {}, {}, {}, {}'.
              format(minlng - margin, minlat - margin, maxlng + margin, maxlat + margin))
    ms.append('set-print-bounds-geo {}, {}, {}, {}'.format(minlng, minlat, maxlng, maxlat))
    ms.append('zoom-map-scale 19')
    dest_filename = '{}\\{}_osm.jpg'.format(dest_labels_rasters, filename)
    ms.append('export-bitmap world-file=true scale={scale} file={dest}'.
              format(scale=scale, dest=dest_filename))

    # Define temporary Maperitive script file
    temp_script_file_name = os.path.join(Path(maperitive_binary).parent, 'maperitive_script.txt')
    print('Running Maperitive script: {}'.format(temp_script_file_name))

    # Write to the Maperitive script used in the subprocess call
    with open(temp_script_file_name, 'w') as maperitive_script_file:
        for a_line in ms:
            maperitive_script_file.write(a_line + '\n')
    maperitive_script_file.close()

    # Call Maperitive and hide the output by redirecting to /dev/null
    subprocess.call(maperitive_binary + ' -exitafter ' + temp_script_file_name, shell=True,
                    stdout=open(os.devnull, 'wb'),
                    stderr=open(os.devnull, 'wb'))

    return dest_filename, filename


def create_osm_file(extent, headers, margin=0.003):
    # The margin is used for bounding box in order to let Maperitive take into consideration marginal elements in image
    s, w, n, e = extent[2] - margin, extent[0] - margin, extent[3] + margin, extent[1] + margin

    # Create query for OverPass
    query = """
          [out:xml]/*fixed by auto repair*/[timeout:25];
          (
          node({}, {}, {}, {});
          <;
          );
          out meta;
          """.format(s, w, n, e)
    data = {
        'data': query
    }

    # Query OverPass to get XML formatted code based on the required coordinates in degrees
    response = requests.post('https://maps.mail.ru/osm/tools/overpass/api/interpreter', headers=headers, data=data)

    # Get the response into an XML formatted .osm file
    osm_file_path = 'map.osm'
    with open(osm_file_path, 'w', encoding="utf-8") as f:
        f.write(response.text)

    return osm_file_path, margin


def get_satellite_image(access_token, extent, raster_image):
    # Get width and height for satellite image
    w, h = raster_image.width, raster_image.height
    # Choose denominator in order to have max zoom for every drone altitude
    sat_w = round(w / 2.7)
    sat_h = round(h / 2.7)

    # Get satellite image based on extent and osm aspect ratio and save image
    r = requests.get(
        'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/[{},{},{},{}]/{}x{}@2x?access_token={}'.format(
            extent[0], extent[2], extent[1], extent[3], sat_w, sat_h, access_token), stream=True)

    if r.status_code == 200:
        with open('temp_sat.png', 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        print("Error at satellite request from Mapbox")

    img_sat = Image.open('temp_sat.png')
    return img_sat


def threshold_img(img_hsv):
    """
    Function for color thresholding and categorical conversion, based on HSV colorspace.
    :param img_hsv: Image in HSV colorspace
    :return: Categorical image
    """

    # Define range of blue color in HSV
    lower_blue = np.array([115, 50, 50])
    upper_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # Define range of green color in HSV
    lower_green = np.array([55, 50, 50])
    upper_green = np.array([65, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    # Define range for red lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([5, 255, 255])
    mask0_red = cv2.inRange(img_hsv, lower_red, upper_red)

    # Define range for red upper mask (170-180)
    lower_red = np.array([175, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1_red = cv2.inRange(img_hsv, lower_red, upper_red)

    # Join red masks
    mask_red = mask0_red + mask1_red

    # Create categorical output based on mask
    output_hsv = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))
    output_hsv[np.where(mask_red == 255)] = 1
    output_hsv[np.where(mask_blue == 255)] = 2
    output_hsv[np.where(mask_green == 255)] = 3

    return output_hsv.astype(np.uint8)


def get_rasters_labels_satellites():
    """
    Main function that can compute rasters, maperitive labels or satellite images according to CLI arguments.
    """
    imgs, dest, scale, maperitive_binary, rulefile, headers,\
        osm_img, raster, label_out, satellite, access_token = labeling_setup()

    nr_jpg_imgs = len(imgs)
    for i in range(nr_jpg_imgs):
        # Get internal and external bounding boxes coordinates
        img_name = imgs[i].img_name
        extent, outer_earth_meters, inner_earth_meters, pic, _ = get_bboxes(imgs[i])

        if raster:
            # Get raster image
            im_raster = map_query(extent, osm_img)
            plt.close()
            # Rotate and crop image
            transform_raster = rot_and_crop(inner_earth_meters, outer_earth_meters, im_raster)
            # Transform and save images
            raster = im_raster.transform((pic.size[0], pic.size[1]), ImageTransform.QuadTransform(transform_raster))
            raster = raster.convert('RGB')
            raster.save(os.path.join(dest, img_name + '_raster.jpg'))

        if label_out:
            # Create osm file used for Maperitive and call subprocess
            source, margin = create_osm_file(extent, headers)
            temp_osm_label, img_name = call_maperitive(extent, source, rulefile, dest, scale,
                                                       maperitive_binary, img_name, margin)
            # Get label image
            # 5482x6482
            im_label = Image.open(temp_osm_label)
            # Rotate and crop image
            transform_label = rot_and_crop(inner_earth_meters, outer_earth_meters, im_label)
            # Transform and save images 5280x3956
            label = im_label.transform((pic.size[0], pic.size[1]), ImageTransform.QuadTransform(transform_label))
            # 5280x3956
            label = np.array(label.convert('RGB'))

            # Threshold image
            label = cv2.cvtColor(label[:, :, ::-1], cv2.COLOR_BGR2HSV)
            thresh_label = threshold_img(label)
            seg_img = Image.fromarray(thresh_label).convert('P')
            seg_img.putpalette(np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=np.uint8))
            seg_img.save(os.path.join(dest, img_name + '_maperitive.png'))

            # Delete unwanted files
            if os.path.isfile(source):
                os.remove(source)

        if satellite:
            # Get satellite image
            im_raster = map_query(extent, osm_img)
            plt.close()
            im_satellite = get_satellite_image(access_token, extent, im_raster)
            # Rotate and crop
            transform_satellite = rot_and_crop(inner_earth_meters, outer_earth_meters, im_satellite)
            # Transform and save image
            sat = im_satellite.transform((pic.size[0], pic.size[1]), ImageTransform.QuadTransform(transform_satellite))
            sat.save(os.path.join(dest, img_name + '_satellite.jpg'))
            # Delete unwanted files
            if os.path.isfile('temp_sat.png'):
                os.remove('temp_sat.png')

        # Delete any unnecessary files (if) created by Maperitive or Mapbox
        for f in os.listdir(dest):
            if 'osm' in f:
                os.remove(os.path.join(dest, f))

        print("Labeling image {} done!".format(img_name))


if __name__ == "__main__":
    get_rasters_labels_satellites()
