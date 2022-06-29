import model_eval_func
import constant_values
import iou_score_metric
import segmentation_models as sm

model_eval_func.model_inference("/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/experiment1/resnet18/experiment_2/trial_2_model",
                                {"my_iou_metric": iou_score_metric.my_iou_metric,
                                 "dice_loss": sm.losses.dice_loss},
                                [constant_values.north_tubbs_imper_lidar["positive"]],
                                "/home/jchavez/prediction/north_tubb_prediction/diceloss/positive/")


model_eval_func.model_inference("/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/experiment1/resnet18/experiment_2/trial_2_model",
                                {"my_iou_metric": iou_score_metric.my_iou_metric,
                                 "dice_loss": sm.losses.dice_loss},
                                [constant_values.north_tubbs_imper_lidar["negative"]],
                                "/home/jchavez/prediction/north_tubb_prediction/diceloss/negative/")
