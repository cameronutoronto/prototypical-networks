#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import random

LAYER_SIZE = 3921#(1 + 28*28) * 5
NUM_HIDDEN = 512 # 1024 gave 74.074 and 1.53687
NUM_CLASSES = 3
TRAIN_SET_SIZE = 40
TRAIN_STEPS = 500
TRAIN_ITS = 5
EVALUATE_ONLY = False

random.seed(1234)

LAYERS = [LAYER_SIZE, LAYER_SIZE]

DROPOUT = 0.5
REG_STRENGTHS = [0.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

#tf.logging.set_verbosity(tf.logging.INFO)

def get_data():
    data = []
    labels = []
    with open("change_in_losses.txt") as f:
      for line in f:
        if (line == '\n'):
          continue
        sline = line[:-1].split(',')
        print(sline[0], sline[783], sline[1568], sline[2352], sline[3136],sline[3920], sline[-1])
        for x in range(len(sline)):
          sline[x] = float(sline[x])
        data.append(sline[:-1])
        labels.append(sline[-1])
    split = int(0.8*len(data))
    train_data = data[:split]
    train_labels = labels[:split]
    eval_data = data[split:]
    eval_labels = labels[split:]
            
    return (np.array(train_data), np.array(train_labels),
                np.array(eval_data), np.array(eval_labels))

def nn_model_fn(features, labels, mode):
  """Model function for NN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, MFCC_NUM_F * MFCC_NUM_C])


  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=input_layer, units=NUM_HIDDEN, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dense, units=NUM_HIDDEN/2, activation=tf.nn.relu)

  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense3 = tf.layers.dense(inputs=dense, units=NUM_HIDDEN/4, activation=tf.nn.relu)

  dropout3 = tf.layers.dropout(
      inputs=dense3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout3, units=NUM_CLASSES)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  print("Loading Data...")
  train_data, train_labels, eval_data, eval_labels = get_data()

  # Create the Estimator
  print("Creating Classifier...")
  input_layer = [tf.feature_column.numeric_column("x", shape=[LAYER_SIZE])]

  tf_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                  l2_regularization_strength=0.0)
  mod_dir = "active_model2"

  voicerecog_classifier = tf.estimator.DNNRegressor(feature_columns=input_layer,
                          hidden_units=LAYERS, model_dir=mod_dir,
                                      optimizer=tf_optimizer,  dropout=DROPOUT)
  tensors_to_log = {}
  logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  for its in range(TRAIN_ITS):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    voicerecog_classifier.train(
        input_fn=train_input_fn,
        steps=TRAIN_STEPS,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = voicerecog_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
