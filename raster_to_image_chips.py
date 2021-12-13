"""
import os
import arcpy
from arcpy import sa
from arcpy import ia


def create_dataset_from_saved_raster(source_raster_layers, source_class, class_val, target_directory, output_nofeature_tiles):
    for raster in source_raster_layers:
        create_image_chips_from_saved_raster(raster, source_class, class_val, target_directory, output_nofeature_tiles)
    return


def create_dataset_from_scratch_m1(source_raster_layers, source_class, target_directory, save_raster, save_names, output_nofeature_tiles):
    for raster in source_raster_layers:
        create_image_chips_from_scratch(source_raster=raster,
                                        source_class=source_class,
                                        source_class_value="Class_Val",
                                        bands=[1, 2, 3],
                                        target_directory=target_directory,
                                        save_raster=False,
                                        save_raster_directory="",
                                        save_raster_name="",
                                        output_nofeature_tiles=output_nofeature_tiles)

    return


def create_dataset_from_scratch_m2(source_raster_layers, source_class, source_class_value, target_directory, output_nofeature_tiles):
    for raster in source_raster_layers:
        create_image_chips_from_scratch(source_raster=raster,
                                        source_class=source_class,
                                        source_class_value=source_class_value,
                                        bands=[4, 1, 2],
                                        target_directory=target_directory,
                                        save_raster=False,
                                        save_raster_directory="",
                                        save_raster_name="",
                                        output_nofeature_tiles=output_nofeature_tiles)
    return


def create_image_chips_from_saved_raster(source_raster, source_class, cv_field, target_directory,
                                         output_nofeature_tiles):
    # creating image chips
    raster_layer = sa.Raster(source_raster)
    arcpy.env.extent = raster_layer
    arcpy.env.cellSize = raster_layer
    arcpy.env.overwriteOutput = 1

    ia.ExportTrainingDataForDeepLearning(in_raster=raster_layer,
                                         out_folder=target_directory,
                                         in_class_data=source_class,
                                         image_chip_format="PNG",
                                         tile_size_x="512",
                                         tile_size_y="512",
                                         stride_x="256",
                                         stride_y="256",
                                         metadata_format="Classified_Tiles",
                                         start_index=0,
                                         class_value_field=cv_field,
                                         rotation_angle=0,
                                         output_nofeature_tiles=output_nofeature_tiles
                                         )
    return


def create_image_chips_from_scratch(source_raster,
                                    source_class,
                                    source_class_value,
                                    bands,
                                    target_directory,
                                    save_raster,
                                    save_raster_directory,
                                    save_raster_name,
                                    output_nofeature_tiles
                                    ):
    # load up the original raster from the source
    original_raster = sa.Raster(source_raster)

    # set environment parameters
    arcpy.env.cellSize = original_raster
    arcpy.env.extent = original_raster
    arcpy.env.overwriteOutput = 1

    # extract the first 3 bands
    raster_3band = sa.ExtractBand(original_raster, bands)  # [x,y,z]
    # remove stretch function
    processed_raster_3band = sa.Stretch(raster=raster_3band, stretch_type="None", gamma=[1, 1, 1])

    if save_raster:
        os.mkdir(save_raster_directory + save_raster_name)
        arcpy.env.workspace = save_raster_directory + save_raster_name
        processed_raster_3band.save(save_raster_name + ".tif")

    # creating image chips
    arcpy.env.extent = processed_raster_3band
    arcpy.env.cellSize = processed_raster_3band
    arcpy.env.workspace = target_directory

    ia.ExportTrainingDataForDeepLearning(in_raster=processed_raster_3band,
                                         out_folder=target_directory,
                                         in_class_data=source_class,
                                         image_chip_format="PNG",
                                         tile_size_x="512",
                                         tile_size_y="512",
                                         stride_x="256",
                                         stride_y="256",
                                         metadata_format="Classified_Tiles",
                                         start_index=0,
                                         class_value_field=source_class_value,
                                         rotation_angle=0,
                                         output_nofeature_tiles=output_nofeature_tiles
                                         )
    return


"""

