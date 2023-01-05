import geopy.distance
import numpy as np


def get_img_size(altitude):
    """
        Input: Altitude in meters
        Output: width and height of image in meters
        FOV and img_ratio for DJI Mavic 3
    """
    FOV = 84  # degrees
    diag = 2 * altitude * np.tan(np.deg2rad(FOV / 2))
    img_ratio = 5280 / 3956  # img is 5280 by 3956 pixels
    width = diag * np.cos(np.arctan(1 / img_ratio))
    height = width / img_ratio
    return width, height


def rot(theta_rad, input_vector):
    mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])
    return (mat @ input_vector.T).T


def get_extent(center_pt, yaw, altitude):
    """
        Inputs:
            center_pt: (lat,long) in decimal degrees of the center of the query
            altitude: drone altitude in meters
        Outputs:
            outer_bbox_earth_deg : extent of the bounding box containing the overlay in degrees
            outer_bbox_earth_meters: extent of the bounding box containing the overlay in meters
            inner_bbox_earth_meters : 4 (x,y) corner points in meters from the center point 
                                                            in the earth coordinate system
    """
    # 5280 / 3956
    width, height = get_img_size(altitude)
    # top left, bottom left, bottom right, top right
    drone_box = np.array(
        [(-width / 2, height / 2), (-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2)])
    rotated = rot(-np.deg2rad(yaw), drone_box)
    extent = [np.min(rotated, axis=0)[0], np.max(rotated, axis=0)[0],
              np.min(rotated, axis=0)[1], np.max(rotated, axis=0)[1]]
    h = extent[3]
    w = extent[1]
    angle = 90 - np.rad2deg(np.arctan(h / w))  # angle with respect to north
    dist = np.sqrt(h ** 2 + w ** 2) / 1000
    v12 = geopy.distance.distance(dist).destination(center_pt, bearing=angle)
    v34 = geopy.distance.distance(dist).destination(center_pt, bearing=angle + 180)
    inner_bbox_drone_meters = drone_box
    inner_bbox_earth_meters = rotated
    outer_bbox_earth_deg = [v34.longitude, v12.longitude, v34.latitude, v12.latitude]
    outer_bbox_earth_meters = extent

    return outer_bbox_earth_deg, outer_bbox_earth_meters, inner_bbox_earth_meters


def shift_coordinate(coord, shift):
    """
        Inputs :
            coord: tuple of x,y coordinates
            shift: tuple vector between new and old origin
        Output : shifted coordinate tuple x,y
    """
    return shift[0] + coord[0], shift[1] + coord[1]


def rot_and_crop(inner_earth_meters, outer_earth_meters, im):
    """
        Inputs :
            inner_earth_meters : 4 (x,y) corner points in meters from the center point 
                                                            in the earth coordinate system
            outer_earth_meters: extent of the bounding box containing the overlay in meters
            im : map image to transform
        Output : transform to rotate and crop the map using the input coordinates
    """
    meter2pix = (im.size[0] / 2) / outer_earth_meters[1]
    inner_earth_pix = list(map(lambda xy: (int(xy[0] * meter2pix), int(xy[1] * meter2pix)), inner_earth_meters))
    shift = (im.size[0] // 2, -im.size[1] // 2)
    bl = shift_coordinate(inner_earth_pix[1], shift)
    br = shift_coordinate(inner_earth_pix[2], shift)
    tl = shift_coordinate(inner_earth_pix[0], shift)
    tr = shift_coordinate(inner_earth_pix[3], shift)
    # top left, bottom left, bottom right, top right
    transform = [*tl, *bl, *br, *tr]
    # negate y values
    transform = [-val if ind % 2 else val for ind, val in enumerate(transform)]
    return transform
