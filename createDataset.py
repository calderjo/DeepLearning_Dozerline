"""
These function will recreate the various ways in which different types of image chips
were made
"""

import os
import arcpy
from arcpy import sa
from arcpy import ia


def create_dataset_method1(source_raster_layers, source_class, target_directory):

    for raster in source_raster_layers:
        create_image_chips_method1(raster, source_class, target_directory, False, "#")

    return


def create_dataset_method2(source_raster_layers, source_class, target_directory):

    for raster in source_raster_layers:
        create_image_chips_method2(raster, source_class, target_directory)

    return


def create_image_chips_method1(source_raster, source_class, target_directory, save_raster, raster_name):
    # load up the original raster from the source
    original_raster = sa.Raster(source_raster)

    arcpy.env.cellSize = original_raster
    arcpy.env.extent = original_raster

    # extract the first 3 bands
    raster_3band = sa.ExtractBand(original_raster, [1, 2, 3])

    # remove stretch function
    processed_raster_3band = sa.Stretch(raster=raster_3band, stretch_type="None", gamma=[1, 1, 1])

    # overwrite
    arcpy.env.overwriteOutput = 1

    if save_raster:
        os.mkdir(source_raster + raster_name)
        arcpy.env.workspace = source_raster + raster_name
        processed_raster_3band.save(raster_name + ".tif")

    # creating image chips
    arcpy.env.extent = processed_raster_3band
    arcpy.env.cellSize = processed_raster_3band
    arcpy.env.overwriteOutput = 1
    cv_field = "Class_Val"

    ia.ExportTrainingDataForDeepLearning(in_raster=processed_raster_3band,
                                         out_folder=target_directory,
                                         in_class_data=source_class,
                                         image_chip_format="PNG",
                                         tile_size_x="512",
                                         tile_size_y="512",
                                         stride_x="512",
                                         stride_y="512",
                                         metadata_format="Classified_Tiles",
                                         start_index=0,
                                         class_value_field=cv_field,
                                         rotation_angle=0
                                         )
    return


def create_image_chips_method2(source_raster, source_class, cv_field, target_directory):

    # creating image chips
    arcpy.env.extent = source_raster
    arcpy.env.cellSize = source_raster
    arcpy.env.overwriteOutput = 1

    ia.ExportTrainingDataForDeepLearning(in_raster=source_raster,
                                         out_folder=target_directory,
                                         in_class_data=source_class,
                                         image_chip_format="PNG",
                                         tile_size_x="512",
                                         tile_size_y="512",
                                         stride_x="512",
                                         stride_y="512",
                                         metadata_format="Classified_Tiles",
                                         start_index=0,
                                         class_value_field=cv_field,
                                         rotation_angle=0
                                         )
    return
