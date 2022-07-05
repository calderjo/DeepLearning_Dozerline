import tensorflow as tf
from tensorflow import keras
import model_pre_processing
import segmentation_models as sm
import model_metrics
import numpy as np
from skimage.measure import label
from skimage import morphology
from osgeo import gdal
import os
from PIL import Image

def model_inference(model_name, image_chips_folder, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

    data_samples = os.listdir(os.path.join(image_chips_folder[0], "images"))
    filtered_data_samples = [samples for samples in data_samples if samples.endswith(".png")]
    filtered_data_samples = sorted(filtered_data_samples)

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'my_iou_metric': model_metrics.my_iou_metric,
            'dice_loss': sm.losses.dice_loss
        }
    )

    # finds all images in the test set
    test_images_label = model_pre_processing.load_data_paths(image_chips_folder)
    test_images = model_pre_processing.load_test_dataset(test_images_label)
    test_images = test_images.batch(32)

    count = 0
    for batch_images, batch_mask in test_images:  # for all the images in test set

        if count == 1:
            break

        batch_predictions = UNET_model.predict(batch_images)  # make a prediction

        for prediction in batch_predictions:  # plot prediction with the input image and ground truth

            name = filtered_data_samples[count]
            image_name = output_directory + str(name)
            gPNG = gdal.Open(os.path.join(image_chips_folder[0], "images", str(name)))

            size = len(image_name)  # text length
            replacement = "tif"  # replace with this
            image_name = image_name.replace(image_name[size - 3:], replacement)

            prediction = np.reshape(prediction, (256, 256))

            output_raster = gdal.GetDriverByName('GTiff').Create(image_name, 256, 256, 1,
                                                                 gdal.GDT_Float32)  # Open the file
            output_raster.SetGeoTransform(gPNG.GetGeoTransform())  # Specify its coordinates
            output_raster.SetProjection(gPNG.GetProjection())  # Exports the coordinate system

            output_raster.GetRasterBand(1).WriteArray(prediction)  # Writes my array to the raster

            output_raster.FlushCache()
            count += 1


def add_pixel_fn(filename: str) -> None:
    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

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


def union_of_dozer_line_images(negative_sample, positive_sample, output_raster):
    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

    tif_files = []

    for path in os.listdir(negative_sample):
        # check if current path is a file
        tif_files.append(os.path.join(negative_sample, path))

    for path in os.listdir(positive_sample):
        # check if current path is a file
        tif_files.append(os.path.join(positive_sample, path))

    gdal.BuildVRT(f'{output_raster}.vrt', tif_files, options=gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0))
    add_pixel_fn(f'{output_raster}.vrt')
    ds = gdal.Open(f'{output_raster}.vrt')
    translate_options = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255"))
    ds = gdal.Translate(f'{output_raster}.tif', ds, options=translate_options)

