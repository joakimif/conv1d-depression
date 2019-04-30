import os
import sys
import random

import numpy as np
import pandas as pd

from tensorflow.keras import backend as K

from cnn.models import ClassificationModel
from cnn.utils import create_segments_and_labels_loo, create_segments_and_labels_madrs_loo

# mode = 'DEPRESSION'
mode = 'DEPRESSION_CLASS'

if mode == 'DEPRESSION':
    create_segments_and_labels = create_segments_and_labels_loo
    n_output_classes = 2
elif mode == 'DEPRESSION_CLASS':
    create_segments_and_labels = create_segments_and_labels_madrs_loo
    n_output_classes = 3

DATASET_DIR = '../datasets'
segment_length = 2880
step = 60
epochs = 10
batch_size = 16
optimizer = 'adam'
verbose = 1
dropout = 0.5

start_i = 0
N = 55

loo_results_filepath = 'leave_one_out_predictions.txt'
loo_filepath = 'leave_one_out.txt'

if os.path.isfile(loo_filepath):
    with open(loo_filepath, 'r') as f:
        start_i = int(f.read())+1

_range = range(start_i, N)

if '--reversed' in sys.argv:
    _range = reversed(_range)

for i in _range:
    print(f'Leaving out participant {i}')

    segments, labels, left_out_segments, left_out_group, input_shape = create_segments_and_labels(DATASET_DIR, segment_length, step, leave_out_id=i)
    model = ClassificationModel(input_shape=input_shape, segment_length=segment_length, step=step, optimizer=optimizer, verbose=verbose, dropout=dropout, n_output_classes=n_output_classes)
    model.fit(segments, labels, batch_size, epochs)
    prediction = model.majority_voting_prediction(left_out_segments)

    with open(loo_results_filepath, 'a') as f:
        f.write(f'{i+1},{left_out_group},{prediction[0]},{prediction[1]},{prediction[2]},{prediction[1]/prediction[2]}\n')

    with open(loo_filepath, 'w') as f:
        f.write(str(i))