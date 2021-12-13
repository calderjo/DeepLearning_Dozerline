import arcpy
import os
import PIL


def create_image_and_labels(source_raster,
                            source_class,
                            source_class_value,
                            bands,
                            target_directory,
                            output_nofeature_tiles
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

    arcpy.ia.ExportTrainingDataForDeepLearning(in_raster=processed_raster_3band,
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


def create_data(target_dir, raster_param, class_params, image_chip_param):

    raster_files = raster_param[0]
    bands = raster_param[1]

    class_file = class_params[0]
    class_value = class_params[1]

    output_nofeature_tiles = image_chip_param

    for raster in raster_files:
        create_image_and_labels(source_raster=raster,
                                source_class=class_file,
                                source_class_value=class_value,
                                bands=bands,
                                target_directory=target_dir,
                                output_nofeature_tiles=output_nofeature_tiles
                                )
    return


def file_processing(file_directory):  # removes excess junk and resolves missing tiles
    remove_junk_files(os.path.join(file_directory, 'images'))
    remove_junk_files(os.path.join(file_directory, 'labels'))
    add_blank_label(os.path.join(file_directory, 'images'), os.path.join(file_directory, 'labels'))


def remove_junk_files(file_directory):  # removes file with ext not equal to .png
    for file in os.listdir(file_directory):
        if file.endswith('.pgw') or file.endswith('.png.aux.xml') or file.endswith('.png.ovr'):
            os.remove(os.path.join(file_directory, file))


def add_blank_label(image_directory, label_directory):
    images = os.listdir(image_directory)  # gets the name of files from the image and label tiles
    labels = os.listdir(label_directory)

    for image in images:
        if image not in labels:  # images without dozer line, we will create a blank ground truth tile
            image_label = PIL.Image.new('I;16', (512, 512))
            image_label.save(os.path.join(label_directory, image), 'PNG')


def create_dataset(save_path, raster_param, class_params, image_chip_param):

    train_target_dir = save_path[0]  # where images are saved
    test_target_dir = save_path[1]

    create_data(train_target_dir, raster_param, class_params, image_chip_param)  # creating image and labels
    create_data(test_target_dir, raster_param, class_params, image_chip_param)

    file_processing(train_target_dir)  # we will clean up the folders
    file_processing(test_target_dir)

