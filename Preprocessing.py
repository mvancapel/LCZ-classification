import rasterio
from rasterio.windows import Window
import numpy as np

counter = 0  # Define a global counter variable

def save_window(x_ind, y_ind, window_size, image_no, mask, satellite, n_bands, directory):
    global counter  # Access the global counter variable
    window = Window(x_ind, y_ind, window_size, window_size)

    # get window transforms
    with rasterio.open(mask) as file:
        mask_local = file.read(window=window)
        transform_mask = file.window_transform(window)

    with rasterio.open(satellite) as file:
        sat_local = file.read(window=window)
        transform_sat = file.window_transform(window)

    mask_local = -999 if mask_local.size==0 else mask_local

    #max = np.max(mask_local)
    min = np.min(mask_local)
    #print(max)
    #print(type(mask_local))
    # count_1 = np.count_nonzero(mask_local == 1)
    # count_2 = np.count_nonzero(mask_local == 2)


    if hasattr(mask_local, "__len__"):
        if not (min == -999.0):
            counter += 1  # Increment the global counter
            i = counter
            print("i = ", i)
            output_mask = f'./{directory}/mask/{i}.tif'
            output_satellite = f'./{directory}/satellite/{i}.tif'
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
    window_size = 64
    x_ind = 0
    y_ind = 0
    stride = 64
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


    satellite_file = r'C:\Users\mavc\Documents\Geomatics\thesis\Data QGIS project\Bigger extent 2022 and 2023\QGIS layers\Stack peaks 1.vrt'
    labels_file = r'C:\Users\mavc\Documents\Geomatics\thesis\Data QGIS project\Bigger extent 2022 and 2023\QGIS layers\ISODATA 12 classes 50 iterations.tif'

    directory = './data_peaks'  # created large mask raster


    # set parameters
    n_bands = 4
    size = 64


    # # create training dataset (cover entire raster)
    create_dataset(satellite_file, labels_file, directory, n_bands)