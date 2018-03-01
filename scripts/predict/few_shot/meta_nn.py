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

LAYER_SIZE = (1 + 8*8) * 5 + 8*8 + 1 #
NUM_HIDDEN = 512 #
NUM_CLASSES = 3
TRAIN_SET_SIZE = 40
TRAIN_STEPS = 500
TRAIN_ITS = 20
EVALUATE_ONLY = False
MODEL_DIR = "active_model3"

random.seed(1234)

LAYERS = [LAYER_SIZE*8, LAYER_SIZE*8]

DROPOUT = 0.3
REG_STRENGTHS = [0.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

tf.logging.set_verbosity(tf.logging.ERROR)

def get_data():
    data = []
    labels = []
    with open("change_in_losses.txt") as f:
      for line in f:
        if (line == '\n'):
          continue
        sline = line[:-1].split(',')

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

def main(unused_argv):
  # Load training and eval data
  print("Loading Data...")
  train_data, train_labels, eval_data, eval_labels = get_data()

  # Create the Estimator
  print("Creating Classifier...")
  input_layer = [tf.feature_column.numeric_column("x", shape=[LAYER_SIZE])]

  tf_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                  l2_regularization_strength=0.0)

  voicerecog_classifier = tf.estimator.DNNRegressor(feature_columns=input_layer,
                          hidden_units=LAYERS, model_dir=MODEL_DIR,
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
