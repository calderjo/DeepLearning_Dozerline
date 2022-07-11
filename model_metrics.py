import numpy as np
import tensorflow as tf


def get_iou_vector(a, b):
    # Numpy version
    batch_size = a.shape[0]
    metric = 0.0

    for batch in range(batch_size):
        t, p = a[batch], b[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        metric += iou

    # take the average over all images in batch
    metric /= batch_size
    return metric

def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.numpy_function(get_iou_vector, [label, pred > 0.5], tf.float64)
