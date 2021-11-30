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


def create_dozer_line_image_chips_bands_irg(
        train_path,
        test_path,
        train_raster_layers,
        test_raster_layers,
        class_layer,
        class_val
):

    raster_to_image_chips.create_dataset_from_scratch_m2(source_raster_layers=train_raster_layers,
                                                         source_class=class_layer,
                                                         source_class_value=class_val,
                                                         target_directory=train_path)

    raster_to_image_chips.create_dataset_from_scratch_m2(source_raster_layers=test_raster_layers,
                                                         source_class=class_layer,
                                                         source_class_value=class_val,
                                                         target_directory=test_path)

    return 0


create_dozer_line_image_chips_bands_irg(

    train_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/bands_IRG_dozer_line/train",

    test_path="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineImageChips/bands_IRG_dozer_line/test",

    train_raster_layers=[
        "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLine_raster/postFireImagery_2017/Nunns Fire North June 14 "
        "2018 1 Foot Imagery/nunns_north_june_14_2018.tif",

        "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLine_raster/postFireImagery_2017/Pocket Fire June 14 2018 1 "
        "Foot Imagery/pocket_june_14_2018.tif",

        "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLine_raster/postFireImagery_2017/Nunns Fire South June 14 "
        "2018 1 Foot Imagery/nunns_south_june_14_2018.tif",

        "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLine_raster/postFireImagery_2017/Tubbs Fire South June 14 "
        "2018 1 Foot Imagery/tubbs_south_june_14_2018.tif"],

    test_raster_layers=[
        "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLine_raster/postFireImagery_2017/Tubbs Fire North June 14 "
        "2018 1 Foot Imagery/tubbs_north_june_14_2018.tif"],

    class_layer="C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineLocation/Corrected_2017_Polygonal_Dozerlines"
                ".gdb/Canopy_Corrected_Dozerline_Data",

    class_val="Class_Val"
)

"""
create_dozer_line_image_chips_via_saved_raster_layers(

    "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/train",
    
    "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset_dozer_line/test",
    
    
    ["C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedNorthNunns.tif",
    
     "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedPocket.tif",
    
     "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedSouthNunns.tif",
     
     "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedSouthTubbs.tif"
     ],
     
     
    ["C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedNorthTubbs.tif"],
    
    
    "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLineLocation/Corrected_2017_Polygonal_Dozerlines.gdb"
    "/Canopy_Corrected_Dozerline_Data",
    
    
    "Class_Val"
)
"""
