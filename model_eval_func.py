import tensorflow as tf
from tensorflow import keras
import dataset_functions
import segmentation_models as sm
import iou_score_metric
import os
from osgeo import gdal
import numpy as np
from skimage.measure import label


def entire_region_evaluate(model_name, custom_objects, test_folders_pos, test_folders_neg, batch_size):

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects=custom_objects
    )

    UNET_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=sm.losses.bce_dice_loss,
                       metrics=[iou_score_metric.my_iou_metric])

    test_sample_paths_com = dataset_functions.load_data_paths([test_folders_neg[0], test_folders_pos[0]])
    test_com_image_data = dataset_functions.load_test_dataset(
        test_sample_paths_com)  # apply pre-processing for resnet 50

    test_com_image_data = test_com_image_data.batch(batch_size)  # same as training
    results_com = UNET_model.evaluate(x=test_com_image_data, batch_size=batch_size)

    test_sample_paths_neg = dataset_functions.load_data_paths(test_folders_neg)
    test_neg_image_data = dataset_functions.load_test_dataset(
        test_sample_paths_neg)  # apply pre-processing for resnet 50

    test_neg_image_data = test_neg_image_data.batch(batch_size)  # same as training
    results_neg = UNET_model.evaluate(x=test_neg_image_data, batch_size=batch_size)

    test_sample_paths_pos = dataset_functions.load_data_paths(test_folders_pos)
    test_pos_image_data = dataset_functions.load_test_dataset(
        test_sample_paths_pos)  # apply pre-processing for resnet 50

    test_pos_image_data = test_pos_image_data.batch(batch_size)  # same as training
    results_pos = UNET_model.evaluate(x=test_pos_image_data, batch_size=batch_size)

    print(f"testing: {str(model_name)} \n\n")

    print(f"Non Dirt Road Sample \n neg_loss: {str(results_neg[0])} \n neg_IOU_score: {str(results_neg[1])}")

    print(f"Dirt Road Sample \n poss_loss: {str(results_pos[0])} \n poss_IOU_score: {str(results_pos[1])}")

    print(f"Combined Samples \n com_loss: {str(results_com[0])} \n com_IOU_score: {str(results_com[1])}")

    print("\n\n Test Finished")


def model_inference(model_name, image_chips_folder, save_path):
    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

    data_samples = os.listdir(os.path.join(image_chips_folder[0], "images"))
    filtered_data_samples = [samples for samples in data_samples if samples.endswith(".png")]
    filtered_data_samples = sorted(filtered_data_samples)

    UNET_model = keras.models.load_model(
        model_name,
        custom_objects={
            'my_iou_metric': iou_score_metric.my_iou_metric,
            'dice_loss': sm.losses.dice_loss
        }
    )

    # finds all images in the test set
    test_images_label = dataset_functions.load_data_paths(image_chips_folder)
    test_images = dataset_functions.load_test_dataset(test_images_label)
    test_images = test_images.batch(32)

    count = 0
    for batch_images, batch_mask in test_images:  # for all the images in test set

        if count == 1:
            break

        batch_predictions = UNET_model.predict(batch_images)  # make a prediction

        for prediction in batch_predictions:  # plot prediction with the input image and ground truth

            name = filtered_data_samples[count]
            image_name = save_path + str(name)
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


def union_of_dozer_line_images(negative_sample, positive_sample, target_file):
    os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
    os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'

    tif_files = []

    for path in os.listdir(negative_sample):
        # check if current path is a file
        tif_files.append(os.path.join(negative_sample, path))

    for path in os.listdir(positive_sample):
        # check if current path is a file
        tif_files.append(os.path.join(positive_sample, path))

    gdal.BuildVRT(f'{target_file}.vrt', tif_files, options=gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0))
    add_pixel_fn(f'{target_file}.vrt')
    ds = gdal.Open(f'{target_file}.vrt')
    translate_options = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255"))
    ds = gdal.Translate(f'{target_file}.tif', ds, options=translate_options)