import arcpy.management
import os


def create_prediction_raster():

    file_list_negative = os.listdir('C:/Users/jonat/Documents/prediction/lidar_model/North_Tubbs/negative')
    for file in file_list_negative:
        if not (file.endswith(".png")):
            file_list_negative.remove(file)

    print("I am done with listing negative files")

    file_list_positive = os.listdir('C:/Users/jonat/Documents/prediction/lidar_model/North_Tubbs/positive')
    for file in file_list_positive:
        if not (file.endswith(".png")):
            file_list_positive.remove(file)

    print("I am done with listing positive files")

    combine_files = file_list_negative + file_list_positive
    output_location = "C:/Users/jonat/Documents/prediction/lidar_model/North_Tubbs"
    raster_dataset_name_with_extension = "predictions_north_tubbs"
    coordinate_system_for_the_raster = "NAD 1983 (2011) StatePlane California II FIPS 0402 (US Feet)"
    pixel_type = "8_BIT_UNSIGNED"
    cellsize = 1
    number_of_bands = 4
    mosaic_method = "FIRST"
    mosaic_colormap_mode = "FIRST"

    arcpy.management.MosaicToNewRaster(input_rasters=combine_files,
                                       output_location=output_location,
                                       raster_dataset_name_with_extension=raster_dataset_name_with_extension,
                                       number_of_bands=number_of_bands)


create_prediction_raster()
