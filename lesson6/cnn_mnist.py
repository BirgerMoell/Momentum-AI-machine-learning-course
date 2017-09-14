from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
	tf.app.run()

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	# Convulutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same"
		activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size[2, 2], strides=2)

# Convolutional layer # 2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
	inputs=pool1,
	filters=64,
	kernel_size=[5, 5],
	padding="same",
	activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
	inputs=dense, rate=0.4, training=mode ==tf.estimator.ModeKeys.TRAIN)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
	# Generate predictions (for PREDICT and EVAL mode)
	"classes": tf.argmax(inout=logits, axis=1),
	# Add softmax_tensor to the graph. It is used for PREDICT and by the
	# 'logging_hook'.
	"probabilities": tf.nn.softmax(logits, name="softmax_tensor#)
}
}
