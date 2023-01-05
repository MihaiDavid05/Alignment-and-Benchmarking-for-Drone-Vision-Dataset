import io
from urllib.request import urlopen, Request

import matplotlib.pyplot as plt
from PIL import Image


def image_spoof(self, tile):  # this function pretends not to be a Python script
    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header('User-agent', 'Anaconda 3')  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = Image.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), 'lower'  # reformat for cartopy


def map_query(extent, osm_img):
    # query extent on cartopy and return map image
    fig = plt.figure(figsize=(5280 / 3956 * 30, 30))  # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs)  # project using coordinate reference system (CRS) of street map
    ax1.set_extent(extent)  # set extents
    scale = 19  # scale cannot be larger than 19 (resolution)
    ax1.add_image(osm_img, int(scale))  # add OSM with resolution specification
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)


def plot_and_save_image(img, filename='test_map.jpg', save=False):
    # plot image without axis and save
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.close()
