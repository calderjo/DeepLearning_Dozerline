from osgeo import gdal
import os

os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

def add_pixel_fn(filename: str) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    header = """  <VRTRasterBand dataType="Byte" band="1" subClass="VRTDerivedRasterBand">"""
    contents = """
    <PixelFunctionType>average</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[
from numba import jit
import numpy as np
@jit(nogil=True)
def average_jit(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt):
    np.mean(in_ar, axis = 0,out = out_ar, dtype = 'uint8')

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt):
    average_jit(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt)
]]>
    </PixelFunctionCode>"""


neg = "/home/jchavez/prediction/north_tubb_prediction/diceloss/negative/"
pos = "/home/jchavez/prediction/north_tubb_prediction/diceloss/positive/"

tifs = []

for path in os.listdir(neg):
    # check if current path is a file
    tifs.append(os.path.join(neg, path))

for path in os.listdir(pos):
    # check if current path is a file
    tifs.append(os.path.join(pos, path))


out_name = "dice_loss_north_tubbs"

gdal.BuildVRT(f'{out_name}.vrt', tifs, options=gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0))
add_pixel_fn(f'{out_name}.vrt')
ds = gdal.Open(f'{out_name}.vrt')
translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 0"))
ds = gdal.Translate(f'{out_name}.tif', ds, options=translateoptions)
