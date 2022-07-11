import numpy as np
from skimage.measure import label
from skimage import morphology
from osgeo import gdal
import os
from PIL import Image


def remove_small_objects(input_raster, output_raster, min_size, connectivity):
    Image.MAX_IMAGE_PIXELS = 2000000000
    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

    img = Image.open(input_raster)
    img_arr = np.array(img)

    [row, col] = img_arr.shape

    noise_reduced = morphology.remove_small_objects(label(img_arr), min_size=min_size, connectivity=connectivity)
    noise_reduced[noise_reduced >= 1] = 1  # removes labels

    gPNG = gdal.Open(input_raster)

    output_raster = gdal.GetDriverByName('GTiff').Create(output_raster, col, row, 1, gdal.GDT_Float32)  # Open the file

    output_raster.SetGeoTransform(gPNG.GetGeoTransform())  # Specify its coordinates
    output_raster.SetProjection(gPNG.GetProjection())  # Exports the coordinate system
    output_raster.GetRasterBand(1).WriteArray(noise_reduced)  # Writes my array to the raster

    output_raster.FlushCache()