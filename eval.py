import model_eval_func

# model_eval_func.union_of_dozer_line_images(r"/home/jchavez/prediction/north_tubb_prediction/diceloss/negative",
#                                            r"/home/jchavez/prediction/north_tubb_prediction/diceloss/positive",
#                                            "output_dice_loss2"
#                                            )













"""
evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_0_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_1_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_2_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_3_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_4_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_5_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_6_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_7_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_8_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
    model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_1/trial_9_model",
    test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
    test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
    batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_2/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet18/experiment_3/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_1/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_2/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet34/experiment_3/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_1/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_2/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet50/experiment_3/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_0_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_1_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_2_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_3_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_4_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_5_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_6_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_7_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_8_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
   model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_1/trial_9_model",
   test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
   test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
   batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_2/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_0_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_1_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_2_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_3_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_4_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_5_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_6_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_7_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_8_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)

evaluate_model(
  model_name="/home/jchavez/model/dozerline_extractor/unet/dataset_lidar/resnet101/experiment_3/trial_9_model",
  test_folders_neg=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Negative_Samples/"],
  test_folders_pos=["/home/jchavez/dataset/Lidar_Dirt_Gravel_Roads/North_Tubbs/Positive_Samples/"],
  batch_size=32
)
"""
