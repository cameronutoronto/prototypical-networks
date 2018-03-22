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

LAYER_SIZE = 4 * 5 + 95 + 1
NUM_HIDDEN = 512 #
NUM_CLASSES = 3
TRAIN_SET_FRAC = 0.8
TRAIN_STEPS = 500
TRAIN_ITS = 20000
EVALUATE_ONLY = False
MODEL_DIR = "active_model_new_0"

random.seed(1234)

LAYERS = [LAYER_SIZE, LAYER_SIZE]

DROPOUT = 0.2
REG_STRENGTHS = [0.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

tf.logging.set_verbosity(tf.logging.ERROR)

ADD_PROTOS = False
ADD_KMEANS_CENTERS = False
ADD_PROTO_DISTS = True
ADD_PROTO_PROBS = True
ADD_KMEANS_CENTERS_DISTS = True
ADD_KMEANS_CENTERS_PROBS = True
ADD_POOL_POINT_DISTS = True
ADD_BASE_LOSS = True


def get_data():
    data = []
    labels = []
    with open("change_in_losses2.txt") as f:
      for line in f:
        if (line == '\n'):
          continue
        sline = line[:-1].split(',')

        for x in range(len(sline)):
          sline[x] = float(sline[x])
        vals = []
        
        start_ind = 0
        end_ind = start_ind + 5*(8*8+1)
        if ADD_PROTOS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])
                
        start_ind = end_ind
        end_ind = start_ind + 5*8*8
        if ADD_KMEANS_CENTERS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])

        start_ind = end_ind + 1 #skip extra base_loss
        end_ind = start_ind + 5
        if ADD_PROTO_DISTS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])

        start_ind = end_ind
        end_ind = start_ind + 5
        if ADD_PROTO_PROBS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])

        start_ind = end_ind
        end_ind = start_ind + 5
        if ADD_KMEANS_CENTERS_DISTS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])

        start_ind = end_ind
        end_ind = start_ind + 5
        if ADD_KMEANS_CENTERS_PROBS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])

        start_ind = end_ind
        end_ind = start_ind + 95
        if ADD_POOL_POINT_DISTS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])

        start_ind = end_ind
        end_ind = start_ind + 1
        if ADD_BASE_LOSS:
          for x in range(start_ind, end_ind):
            vals.append(sline[x])
            
        data.append(vals)
        labels.append(sline[-1])
    split = int(TRAIN_SET_FRAC*len(data))
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
  input_layer = [tf.feature_column.numeric_column("x", shape=[len(train_data[0])])]

  tf_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                  l2_regularization_strength=0.0)
  tf_optimizer = tf.train.AdamOptimizer()

  meta_nn = tf.estimator.DNNRegressor(feature_columns=input_layer,
                          hidden_units=LAYERS, model_dir=MODEL_DIR,
                                      optimizer=tf_optimizer,  dropout=DROPOUT) #mean squred error L2 loss
  tensors_to_log = {}
  logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  for its in range(TRAIN_ITS):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100000,
        num_epochs=None,
        shuffle=True)

    meta_nn.train(
        input_fn=train_input_fn,
        steps=TRAIN_STEPS,
        hooks=[logging_hook])

    # Evaluate training set
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = meta_nn.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = meta_nn.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
