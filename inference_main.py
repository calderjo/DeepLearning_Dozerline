import os.path
import model_inference
import model_post_processing
import model_metrics
import segmentation_models as sm
import dataset_paths

def inference(model_path, costume_object, positive_samples, negative_samples, output_directory):

    model_inference.model_inference(
        model_path,
        costume_object,
        [positive_samples,negative_samples],
        os.path.join(output_directory, "samples/"))

    model_inference.union_of_dozer_line_images(
        os.path.join(output_directory, "samples/"),
        os.path.join(output_directory, "map_wide_view"))


inference(
    "/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/experiment1/resnet18/experiment_2/trial_2_model",
    {"my_iou_metric": model_metrics.my_iou_metric,"dice_loss": sm.losses.dice_loss},
    dataset_paths.north_tubbs_imper_lidar["negative"],
    dataset_paths.north_tubbs_imper_lidar["positive"],
    "/home/jchavez/prediction/north_tubb_prediction/diceloss/experiment_2/trial_2")

