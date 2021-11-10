from arcpy import sa
import raster_to_image_chips


def create_dozer_line_image_chips_via_saved_raster_layers(
        train_path,
        test_path,
        train_raster_layers,
        test_raster_layers,
        class_layer,
        class_val
):
    train_raster_layers = sorted(train_raster_layers)
    test_raster_layers = sorted(test_raster_layers)

    raster_to_image_chips.create_dataset_from_saved_raster(train_raster_layers, class_layer, class_val, train_path)
    raster_to_image_chips.create_dataset_from_saved_raster(test_raster_layers, class_layer, class_val, test_path)

    return


create_dozer_line_image_chips_via_saved_raster_layers(
    "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/train",

    "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/test",

    ["C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedNorthNunns.tif",
     "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedSouthNunns.tif",
     "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedPocket.tif",
     "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedSouthTubbs.tif"],

    ["C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedNorthTubbs.tif"],

    "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineLocation/Corrected_2017_Polygonal_Dozerlines.gdb"
    "/Canopy_Corrected_Dozerline_Data",

    "Class_Val"
)
