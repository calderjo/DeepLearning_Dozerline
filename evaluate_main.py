import model_evaluate

prediction = "/home/jchavez/prediction/north_tubb_prediction/diceloss/experiment_2/trial_2/map_wide_view.tif"
ground_truth = "/home/jchavez/dataset/dozerline/Imper_data_lidar/North_Tubbs/Whole_Dozerline_Nort_1.tif"

model_evaluate.map_wide_based_evaluation(ground_truth, ground_truth)



