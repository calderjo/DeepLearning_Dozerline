import arcpy
import os
import constant_info
import PIL


def create_image_and_labels(source_raster,
                            source_class,
                            source_class_value,
                            bands,
                            target_directory,
                            output_nofeature_tiles,
                            in_mask_poly
                            ):
    # load up the original raster from the source
    original_raster = arcpy.sa.Raster(source_raster)

    # set environment parameters
    arcpy.env.cellSize = original_raster
    arcpy.env.extent = original_raster
    arcpy.env.overwriteOutput = 1

    # extract spec bands
    raster_3band = arcpy.sa.ExtractBand(original_raster, bands)  # [x,y,z]

    # remove stretch function
    processed_raster_3band = arcpy.sa.Stretch(raster=raster_3band, stretch_type="None", gamma=[1, 1, 1])

    # creating image chips
    arcpy.env.extent = processed_raster_3band
    arcpy.env.cellSize = processed_raster_3band
    arcpy.env.workspace = target_directory

    if in_mask_poly == "":
        arcpy.ia.ExportTrainingDataForDeepLearning(in_raster=processed_raster_3band,
                                                   out_folder=target_directory,
                                                   in_class_data=source_class,
                                                   image_chip_format="PNG",
                                                   tile_size_x="256",
                                                   tile_size_y="256",
                                                   stride_x="128",
                                                   stride_y="128",
                                                   metadata_format="Classified_Tiles",
                                                   start_index=0,
                                                   class_value_field=source_class_value,
                                                   rotation_angle=0,
                                                   output_nofeature_tiles=output_nofeature_tiles)
    else:
        arcpy.ia.ExportTrainingDataForDeepLearning(in_raster=processed_raster_3band,
                                                   out_folder=target_directory,
                                                   in_class_data=source_class,
                                                   image_chip_format="PNG",
                                                   tile_size_x="256",
                                                   tile_size_y="256",
                                                   stride_x="128",
                                                   stride_y="128",
                                                   in_mask_polygons=in_mask_poly,
                                                   metadata_format="Classified_Tiles",
                                                   start_index=0,
                                                   class_value_field=source_class_value,
                                                   rotation_angle=0,
                                                   output_nofeature_tiles=output_nofeature_tiles)

    return


def create_data(target_dir, raster_param, class_params, image_chip_param):
    raster = raster_param[0]
    bands = raster_param[1]

    class_file = class_params[0]
    class_value = class_params[1]

    output_nofeature_tiles = image_chip_param[0]
    in_mask_poly = image_chip_param[1]

    create_image_and_labels(source_raster=raster,
                            source_class=class_file,
                            source_class_value=class_value,
                            bands=bands,
                            target_directory=target_dir,
                            output_nofeature_tiles=output_nofeature_tiles,
                            in_mask_poly=in_mask_poly)

    return


def file_processing(positive_samples, negative_samples):  # removes excess junk and resolves missing tiles
    moveNegativeSamples(positive_samples, negative_samples)


def moveNegativeSamples(positive_samples, negative_samples):
    images = os.listdir(os.path.join(positive_samples, 'images'))  # list image names
    labels = os.listdir(os.path.join(positive_samples, 'labels'))  # list label names

    for image in images:
        if image not in labels:
            image_label = PIL.Image.new('I;16', (256, 256))  # images without dozer line, we will create a blank
            image_label.save(os.path.join(negative_samples, "labels", image), 'PNG')
            os.rename(os.path.join(positive_samples, 'images', image), os.path.join(negative_samples, 'images', image))


def create_image_chips(positive_sample_path,
                       negative_sample_path,
                       raster_param,
                       class_params,
                       image_chip_param):

    os.mkdir(positive_sample_path)
    os.mkdir(negative_sample_path)

    os.mkdir(os.path.join(negative_sample_path, "images"))
    os.mkdir(os.path.join(negative_sample_path, "labels"))

    create_data(positive_sample_path, raster_param, class_params, image_chip_param)  # creating image and labels
    file_processing(positive_sample_path, negative_sample_path)  # we will clean up the folders


create_image_chips(
    positive_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/North_Nunns"
                         "/Positive_Samples",

    negative_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/North_Nunns"
                         "/Negative_Samples",

    raster_param=[constant_info.north_nunns["aerial_imagery"], constant_info.infrared],

    class_params=[constant_info.impervious_dirt_roads["file"], constant_info.impervious_dirt_roads["value"]],

    image_chip_param=[False, constant_info.north_nunns["fire_boundary"]]
)


create_image_chips(
    positive_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/South_Nunns"
                         "/Positive_Samples",

    negative_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/South_Nunns"
                         "/Negative_Samples",

    raster_param=[constant_info.south_nunns["aerial_imagery"], constant_info.infrared],

    class_params=[constant_info.impervious_dirt_roads["file"], constant_info.impervious_dirt_roads["value"]],

    image_chip_param=[False, constant_info.south_nunns["fire_boundary"]]
)


create_image_chips(

    positive_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/North_Tubbs"
                         "/Positive_Samples",

    negative_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/North_Tubbs"
                         "/Negative_Samples",

    raster_param=[constant_info.north_tubbs["aerial_imagery"], constant_info.infrared],

    class_params=[constant_info.impervious_dirt_roads["file"], constant_info.impervious_dirt_roads["value"]],

    image_chip_param=[False, constant_info.north_tubbs["fire_boundary"]]
)


create_image_chips(

    positive_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/South_Tubbs"
                         "/Positive_Samples",

    negative_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/South_Tubbs"
                         "/Negative_Samples",

    raster_param=[constant_info.south_tubbs["aerial_imagery"], constant_info.infrared],

    class_params=[constant_info.impervious_dirt_roads["file"], constant_info.impervious_dirt_roads["value"]],

    image_chip_param=[False, constant_info.south_tubbs["fire_boundary"]]
)

create_image_chips(

    positive_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/Pocket"
                         "/Positive_Samples",

    negative_sample_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/Imper_data/Pocket"
                         "/Negative_Samples",

    raster_param=[constant_info.pocket["aerial_imagery"], constant_info.infrared],

    class_params=[constant_info.impervious_dirt_roads["file"], constant_info.impervious_dirt_roads["value"]],

    image_chip_param=[False, constant_info.pocket["fire_boundary"]]
)

