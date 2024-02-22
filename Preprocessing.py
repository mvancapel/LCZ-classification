import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.enums import Resampling

import geopandas as gpd
import os
import glob
import math
import shapely
from rasterio import features
from shapely.geometry import Polygon, LineString, Point, shape
from osgeo import gdal, ogr
import pandas as pd
import numpy as np
from pathlib import Path
import json
from shapely.geometry import box
from fiona.crs import from_epsg
import pycrs

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def save_window(x_ind, y_ind, window_size, image_no, mask, satellite, n_bands, directory):
    output_mask = f'./{directory}/mask/{image_no}.tif'
    output_satellite = f'./{directory}/satellite/{image_no}.tif'
    window = Window(x_ind, y_ind, window_size, window_size)

    # get window transforms
    with rasterio.open(mask) as file:
        mask_local = file.read(window=window)
        transform_mask = file.window_transform(window)

    with rasterio.open(satellite) as file:
        sat_local = file.read(window=window)
        transform_sat = file.window_transform(window)

    mask_local = -999 if mask_local.size==0 else mask_local

    max = np.max(mask_local)
    print(max)
    print(type(mask_local))
    count_1 = np.count_nonzero(mask_local == 1)
    count_2 = np.count_nonzero(mask_local == 2)


    if hasattr(mask_local, "__len__"):
        with rasterio.open(output_mask, 'w',
                            driver='GTiff', width=window_size, height=window_size, count=1,
                            dtype=rasterio.uint8, transform=transform_mask) as dst:
            dst.write(mask_local)

    # Save clip of satellite
        with rasterio.open(output_satellite, 'w',
                            driver='GTiff', width=window_size, height=window_size, count=n_bands,
                            dtype=rasterio.rasterio.float32, transform=transform_sat) as dst:
            dst.write(sat_local)


def create_dataset(satellite, mask, directory, n_bands):
    print('Creating training dataset...')
    image = 0

    open_mask = rasterio.open(mask)
    mask_array = open_mask.read()
    open_satellite = rasterio.open(satellite)
    sat_array = open_satellite.read()

    # set extent to tile (clipped combined bands)
    with open_satellite as src:
        profile = src.profile
        transform = src.transform
        width = src.width
        height = src.height

    resolution = 10
    # compute tile size
    w_px = width
    h_px = height
    shape_tile = w_px, h_px

    # initialize moving window
    window_size = 32
    x_ind = 0
    y_ind = 0
    stride = 32
    print('window initialized')

    # traverse
    while x_ind < (mask_array.shape[2] - window_size - 1):
        # save image
        while y_ind < (mask_array.shape[1] - window_size - 1):
            # save image
            image += 1
            # print(f'window {image}:{x_ind},{y_ind}')
            # move down
            y_ind += stride
            save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)
        save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)
        image += 1
        # print(f'window {image}:{x_ind},{y_ind}')
        # move across
        x_ind += stride
        y_ind = 0
        save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)

    while y_ind < (mask_array.shape[1] - window_size - 1):
        # save image
        image += 1
        # print(f'window {image}:{x_ind},{y_ind}')
        # move down
        y_ind += stride
        save_window(x_ind, y_ind, window_size, image, mask, satellite, n_bands, directory)


    print('Finished creating training dataset!!!')





if __name__ == "__main__":

    data_path = r'C:\Users\mavc\Downloads\Den haag 2022\Selected images\Images with Rotterdam'

    satellite_file = r'C:\Users\mavc\Documents\Geomatics\thesis\Data QGIS project\Rotterdam and Den Haag\202210411504stack_raster.tif'
    labels_file = r'C:\Users\mavc\Documents\Geomatics\thesis\Data QGIS project\Cluster ISODATA 20 classes Den Haag en Rotterdam (50 iterations).tif'

    mask_output = 'labels.tif'  # name to use for created mask raster
    directory = './data'  # created large mask raster

    # to create label raster
    #mask_rasters(satellite_file, polygons_file, mask_output)

    # set parameters
    n_bands = 25
    size = 32

    # grid = create_grid(satellite_file, mask_output)
    scale_factor = 8

    # # create training dataset (along grid)
    # dataset_from_grid(satellite_file, mask_output, directory, n_bands, size, grid)

    # # create training dataset (cover entire raster)
    create_dataset(satellite_file, labels_file, directory, n_bands)