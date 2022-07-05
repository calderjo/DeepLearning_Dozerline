

# neg = "/home/jchavez/prediction/north_tubb_prediction/diceloss/negative/"
#
# pos = "/home/jchavez/prediction/north_tubb_prediction/diceloss/positive/"
#
# output = "output.tif"
#
# model_eval_func.union_of_dozer_line_images(neg, pos, output)
#
# model_eval_func.remove_small_roads(output, "output_small_blobs_removed.tif", 100, 1)
#
# model_eval_func.remove_small_roads(output, "output_small_blobs_removed.tif", 200, 1)
#
# model_eval_func.remove_small_roads(output, "output_small_blobs_removed.tif", 300, 1)
#
# model_eval_func.remove_small_roads(output, "output_small_blobs_removed.tif", 400, 1)
#
# model_eval_func.remove_small_roads(output, "output_small_blobs_removed.tif", 500, 1)
#
# model_eval_func.remove_small_roads(output, "output_small_blobs_removed.tif", 400, 1)
#
# import iou_score_metric
# import segmentation_models as sm
#
# model_eval_func.model_inference(
#     "/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/experiment1/resnet18/experiment_2/trial_2_model",
#     {"my_iou_metric": iou_score_metric.my_iou_metric,
#      "dice_loss": sm.losses.dice_loss}, [constant_values.north_tubbs_imper_lidar["positive"]])
#
# model_eval_func.model_inference(
#     "/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/experiment1/resnet18/experiment_2/trial_2_model",
#     {"my_iou_metric": iou_score_metric.my_iou_metric,
#      "dice_loss": sm.losses.dice_loss}, [constant_values.north_tubbs_imper_lidar["negative"]])
#
# model_eval_func.union_of_dozer_line_images(,










# from osgeo import gdal
# import os
#
# os.environ['PROJ_LIB'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/proj/'
# os.environ['GDAL_DATA'] = '/home/jchavez/miniconda3/envs/deepLearning_dozerLine/share/'
#
# def add_pixel_fn(filename: str) -> None:
#     """inserts pixel-function into vrt file named 'filename'
#     Args:
#         filename (:obj:`string`): name of file, into which the function will be inserted
#         resample_name (:obj:`string`): name of resampling method
#     """
#
#     header = """  <VRTRasterBand dataType="Byte" band="1" subClass="VRTDerivedRasterBand">"""
#     contents = """
#     <PixelFunctionType>average</PixelFunctionType>
#     <PixelFunctionLanguage>Python</PixelFunctionLanguage>
#     <PixelFunctionCode><![CDATA[
# from numba import jit
# import numpy as np
# @jit(nogil=True)
# def average_jit(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt):
#     np.mean(in_ar, axis = 0,out = out_ar, dtype = 'uint8')
#
# def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt):
#     average_jit(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt)
# ]]>
#     </PixelFunctionCode>"""
#
#
# neg = "/home/jchavez/prediction/north_tubb_prediction/diceloss/negative/"
# pos = "/home/jchavez/prediction/north_tubb_prediction/diceloss/positive/"
#
# tifs = []
#
# for path in os.listdir(neg):
#     # check if current path is a file
#     tifs.append(os.path.join(neg, path))
#
# for path in os.listdir(pos):
#     # check if current path is a file
#     tifs.append(os.path.join(pos, path))
#
#
# out_name = "dice_loss_north_tubbs"
#
# gdal.BuildVRT(f'{out_name}.vrt', tifs, options=gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0))
# add_pixel_fn(f'{out_name}.vrt')
# ds = gdal.Open(f'{out_name}.vrt')
# translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 0"))
# ds = gdal.Translate(f'{out_name}.tif', ds, options=translateoptions)
#
# import constant_values
# import train_model
#
# UNET_param = {
#     'batch_size': 32,
#     'epochs': 15,
#     'input_size': (256, 256, 3),
#     'learning_rate': 0.01,
#     'backbone': 'resnet18'
# }
#
# folds = [
#     constant_values.north_nunns_imper_lidar["positive"],
#     constant_values.north_tubbs_imper_lidar["positive"],
#     constant_values.south_nunns_imper_lidar["positive"],
#     constant_values.south_tubbs_imper_lidar["positive"],
#     constant_values.pocket_imper_lidar["positive"]
# ]
#
# fold1_training = [x for i, x in enumerate(folds) if i != 0]
# fold2_training = [x for i, x in enumerate(folds) if i != 1]
# fold3_training = [x for i, x in enumerate(folds) if i != 2]
# fold4_training = [x for i, x in enumerate(folds) if i != 3]
# fold5_training = [x for i, x in enumerate(folds) if i != 4]
#
# training_folds = [fold1_training, fold2_training, fold3_training, fold4_training, fold5_training]
#
# count = 3
# for fold in training_folds[2:]:
#     train_model.train_UNET_RESNET_model(
#         seed=479,
#         training_dirs=fold,
#         unet_params=UNET_param,
#         experiment_target_dir="/home/jchavez/model/dozerline_extractor/unet/lofo/experiment2",
#         trial_number=count
#     )
#     count += 1
