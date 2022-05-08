import os
lidar_base_dir = "/home/jchavez/dataset/dozerline/Imper_data_lidar"

north_tubbs_imper_lidar = {
    "positive": os.path.join(lidar_base_dir, "North_Tubbs/Positive_Samples"),
    "negative": os.path.join(lidar_base_dir, "North_Tubbs/Negative_Samples")
}

south_tubbs_imper_lidar = {
    "positive": os.path.join(lidar_base_dir, "South_Tubbs/Positive_Samples"),
    "negative": os.path.join(lidar_base_dir, "South_Tubbs/Negative_Samples")
}

north_nunns_imper_lidar = {
    "positive": os.path.join(lidar_base_dir, "North_Nunns/Positive_Samples"),
    "negative": os.path.join(lidar_base_dir, "North_Nunns/Negative_Samples")
}

south_nunns_imper_lidar = {
    "positive": os.path.join(lidar_base_dir, "South_Nunns/Positive_Samples"),
    "negative": os.path.join(lidar_base_dir, "South_Nunns/Negative_Samples")
}

pocket_imper_lidar = {
    "positive": os.path.join(lidar_base_dir, "Pocket/Positive_Samples"),
    "negative": os.path.join(lidar_base_dir, "Pocket/Negative_Samples")
}

