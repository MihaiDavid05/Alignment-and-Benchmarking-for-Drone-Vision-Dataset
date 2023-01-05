from dataclasses import dataclass
from os import listdir
from os.path import isfile, join

import tifffile
import xmltodict


def get_image_list(path, img_format='JPG'):
    '''
        input : path to images directory, img_format (default : JPG)
        output: list of paths to images of selected format in directory 
    '''
    images = [f for f in listdir(path) if isfile(join(path, f))]
    paths = [path + '/' + image for image in images if img_format in str(image)]
    paths.sort()
    return paths


def get_dng_jpg(path):
    '''
        input : path to images directory, img_format (default : JPG)
        output: list of image names and two lists of paths to DNG and JPG images in directory 
    '''
    images_dng = get_image_list(path, img_format='DNG')
    images_jpg = get_image_list(path, img_format='JPG')

    # check if each image has two formats
    jpg_names = [img.split('/')[-1][:-4] for img in images_jpg]
    dng_names = [img.split('/')[-1][:-4] for img in images_dng]
    difference1 = (set(dng_names)).difference(set(jpg_names))
    difference2 = (set(jpg_names)).difference(set(dng_names))
    if difference1:
        print(f'Warning, {difference1} is not in both formats')
        return dng_names, images_dng, images_jpg

    elif difference2:
        print(f'Warning, {difference2} is not in both formats')
        return jpg_names, images_dng, images_jpg
    else:
        return jpg_names, images_dng, images_jpg


@dataclass
class DJIImage:
    img_name: str
    dng_path: str
    jpg_path: str
    metadata: dict


def get_imgs(path):
    img_names, images_dng, images_jpg = get_dng_jpg(path)
    if len(img_names) != len(images_dng) or len(img_names) != len(images_jpg):
        raise Exception("The number of dng and jpg files don't match")
    else:
        imgs = []
        for name, dng, jpg in zip(img_names, images_dng, images_jpg):
            metadata = get_dji_metadata(dng)
            img = DJIImage(name, dng, jpg, metadata)
            imgs.append(img)
        return imgs


def get_raw_dji_metadata(img_name):
    """
    Input : path to image
    Output : raw XML metadata
    """
    # read the image data using PIL
    tiffexifdict = dict()
    with tifffile.TiffFile(img_name) as tif:
        for page in tif.pages:
            for tag in page.tags:
                tag_name, tag_value = tag.name, tag.value
                tiffexifdict[tag_name] = tag_value

    dji_metadict = xmltodict.parse(tiffexifdict['XMP'])
    dji_metadict = dji_metadict['x:xmpmeta']['rdf:RDF']['rdf:Description']
    return dji_metadict


def get_dji_metadata(filename):
    """
    Input : image
    Output : clean XML metadata dictionary
    """
    keep_columns = ['@xmp:CreateDate',
                    '@drone-dji:GpsLatitude', '@drone-dji:GpsLongitude',
                    '@drone-dji:AbsoluteAltitude', '@drone-dji:RelativeAltitude',
                    '@drone-dji:GimbalRollDegree', '@drone-dji:GimbalYawDegree',
                    '@drone-dji:GimbalPitchDegree', '@drone-dji:FlightRollDegree',
                    '@drone-dji:FlightYawDegree', '@drone-dji:FlightPitchDegree',
                    '@drone-dji:FlightXSpeed', '@drone-dji:FlightYSpeed',
                    '@drone-dji:FlightZSpeed', '@drone-dji:CamReverse',
                    '@drone-dji:GimbalReverse']

    md = get_raw_dji_metadata(filename)

    dict_filter = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
    md = dict_filter(md, keep_columns)

    float_columns = ['@drone-dji:GpsLatitude', '@drone-dji:GpsLongitude',
                     '@drone-dji:AbsoluteAltitude', '@drone-dji:RelativeAltitude',
                     '@drone-dji:GimbalRollDegree', '@drone-dji:GimbalYawDegree',
                     '@drone-dji:GimbalPitchDegree', '@drone-dji:FlightRollDegree',
                     '@drone-dji:FlightYawDegree', '@drone-dji:FlightPitchDegree',
                     '@drone-dji:FlightXSpeed', '@drone-dji:FlightYSpeed',
                     '@drone-dji:FlightZSpeed', '@drone-dji:CamReverse',
                     '@drone-dji:GimbalReverse']

    for column in float_columns:
        md[column] = float(md[column])

    # change key names
    md = {k.replace('@xmp:', ''): v for k, v in md.items()}
    md = {k.replace('@drone-dji:', ''): v for k, v in md.items()}
    return md
