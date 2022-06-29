from skimage import morphology
from osgeo import gdal
import numpy as np
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = 2000000000
os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

input_raster = "dice_loss_north_tubbs.tif"
output_raster = "remove_blob_65536_60.tif"

img = Image.open(input_raster)
img_arr = np.array(img)

img_arr = morphology.remove_small_objects(img_arr, min_size=65536, connectivity=60)

gPNG = gdal.Open(input_raster)

output_raster = gdal.GetDriverByName('GTiff').Create(output_raster, 38400, 38400, 1, gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(gPNG.GetGeoTransform())  # Specify its coordinates
output_raster.SetProjection(gPNG.GetProjection())  # Exports the coordinate system
output_raster.GetRasterBand(1).WriteArray(img_arr)  # Writes my array to the raster

output_raster.FlushCache()
