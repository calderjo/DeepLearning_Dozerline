import model_evaluate
import sys
import os


def evaluate_main(prediction_map_path, ground_truth_map_path, output_path, output_file_name):

    os.makedirs(output_path, exist_ok=True)

    orig_stdout = sys.stdout
    file = open(os.path.join(output_path, output_file_name), 'w')
    sys.stdout = file

    model_evaluate.map_wide_based_evaluation(prediction_map_path, ground_truth_map_path)

    sys.stdout = orig_stdout
    file.close()

    return


for i in range(10):
    prediction = f"/home/jchavez/prediction/north_tubb_prediction/dice_loss/experiment_2/trial_{i}/map_wide_view.tif"
    ground_truth = "/home/jchavez/dataset/dozerline/Imper_data_lidar_map_wide/North_Tubbs_Map_Wide_Label.tif"
    out_path = f"/home/jchavez/results/dice_loss/experiment_2/trial_{i}/"
    filename = "results.txt"

    evaluate_main(prediction, ground_truth, out_path, filename)




