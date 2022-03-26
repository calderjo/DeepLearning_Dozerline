# Import system modules
import arcpy
from arcpy.ia import *

"""
Usage: ClassifyPixelsUsingDeepLearning(in_raster,out_classified_raster, 
       in_classifier_definition, {arguments}, {processing_mode})

"""

# Set local variables
in_raster = "C:/Users/jonat/Documents/Dataset/DozerLine/DozerLine_raster/postFireImagery_2017/Tubbs Fire North June " \
            "14 2018 " \
            "1 Foot Imagery/tubbs_north_june_14_2018.tif"

in_model_definition = "C:/Users/jonat/Documents/deepLearningModel/dozerlineExtraction/BestModel_from_experiment" \
                      "/model_definition.emd"

model_arguments = "padding 0; batch_size 16"
processing_mode = "PROCESS_ITEMS_SEPARATELY"

# Check out the ArcGIS Image Analyst extension license
arcpy.CheckOutExtension("ImageAnalyst")

# Execute
Out_classified_raster = ClassifyPixelsUsingDeepLearning(in_raster,
                                                        in_model_definition, model_arguments, processing_mode)

Out_classified_raster.save("c:\\classifydata\\classified_moncton.tif")
