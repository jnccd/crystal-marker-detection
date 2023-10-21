import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_loss(y_true, y_pred):
    """
    Hungarian (Munkres) Loss Function

    Arguments:
    y_true -- true target labels, shape (batch_size, num_samples)
    y_pred -- predicted target labels, shape (batch_size, num_samples)

    Returns:
    loss -- Hungarian loss
    """

    batch_size = tf.shape(y_true)[0]
    num_samples = tf.shape(y_true)[1]

    # Modify the predicted probabilities to create a cost matrix with non-negative values
    cost_matrix = 1.0 - y_pred

    # Apply the Hungarian algorithm to find the optimal assignment
    assignment = tf.py_function(func=linear_sum_assignment, inp=[cost_matrix], Tout=tf.int64)

    # Create a one-hot matrix based on the assignment
    assignment_one_hot = tf.one_hot(assignment[0], depth=num_samples)

    # Compute the element-wise product between the one-hot matrix and true labels
    selected_labels = tf.reduce_sum(y_true * assignment_one_hot, axis=1)

    # Calculate the loss as the sum of selected labels (without the negative sign)
    loss = tf.reduce_sum(selected_labels)

    return loss

# Example usage:
# Define true and predicted labels
# true_labels = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
# predicted_labels = tf.constant([[0.2, 0.6, 0.2], [0.4, 0.4, 0.2]], dtype=tf.float32)
true_labels = tf.constant([[1.0, 0.0], [1.0, 0.2]], dtype=tf.float32)
predicted_labels = tf.constant([[0.5, 0.5], [0.0, 0.2]], dtype=tf.float32)

# Compute the Hungarian loss
loss = hungarian_loss(true_labels, predicted_labels)
print("Hungarian Loss:", loss.numpy())