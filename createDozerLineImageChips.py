from arcpy import sa
import createDataset


def create_dozer_line_image_chips_method2():
    # this dir holds images that will be used for the training process
    training_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/train"

    # this dir holds images that will be with held from the training process
    test_img_path = "C:/Users/jonat/Documents/DeeplearningDozerlineNotebook/dataset/test"

    processedNorthNunns = sa.Raster(
        "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedNorthNunns.tif")
    processedSouthNunns = sa.Raster(
        "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedSouthNunns.tif")
    processedNorthTubbs = sa.Raster(
        "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedNorthTubbs.tif")
    processedSouthTubbs = sa.Raster(
        "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedSouthTubbs.tif")
    processedPocket = sa.Raster(
        "C:/Users/jonat/Documents/ArcGIS/Projects/dLearn_dozerLineExtraction/dozerLine_raster/processedPocket.tif")

    source_class = "C:/Users/jonat/Documents/Dataset/DozerLine/" + \
                   "DozerLineLocation/Corrected_2017_Polygonal_Dozerlines.gdb/Canopy_Corrected_Dozerline_Data "

    # training will come from the tubbs S, pocket, nunns N & S
    train_region = [processedNorthNunns, processedSouthNunns, processedPocket, processedSouthTubbs]
    createDataset.create_image_chips_method2(train_region, source_class, "Class_Val", training_img_path)

    # testing will come from the tubbs N
    test_region = [processedNorthTubbs]
    createDataset.create_image_chips_method2(test_region, source_class, "Class_Val", test_img_path)


create_dozer_line_image_chips_method2()
