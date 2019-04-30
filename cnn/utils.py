import os
import random

import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical

MADRS_LABLES = ['Normal', 'Mild', 'Moderate']
MADRS_VALUES = [0, 7, 20]
classes = len(MADRS_VALUES)

"""
Create segments and labels for classification models.

- dataset_dir:      location of dataset files
- segment_length:   length of each segment
- step:             how many indexes to jump after each iteration
- n_output_classes: how many target classes in classification

Returns:
- segments
- labels
- input_shape

"""
def create_segments_and_labels(dataset_dir, segment_length, step, n_output_classes=2):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['afftype'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            segments.append([segment])

            if p['afftype'].values[0] == 0:
                labels.append(0)
            else:
                labels.append(1)

    labels = np.asarray(labels).astype('float32')
    labels = to_categorical(labels, n_output_classes)
    
    segments = np.asarray(segments).reshape(-1, segment_length, 1)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    
    return segments, labels, input_shape

"""
Create segments and labels for MADRS prediction model.

- dataset_dir:      location of dataset files
- segment_length:   length of each segment
- step:             how many indexes to jump after each iteration

Returns:
- segments
- labels
- input_shape

"""
def create_segments_and_labels_prediction(dataset_dir, segment_length, step):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['madrs2'].fillna(0, inplace=True)

    segments = []
    labels = []

    for person in scores['number']:
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        for i in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[i : i + segment_length]
            
            segments.append([segment])
            labels.append(p['madrs2'].values[0])

    segments = np.asarray(segments)
    segments = segments.reshape(-1, segment_length, 1)

    input_shape = segments.shape[1]
    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    labels = np.asarray(labels).astype('float32')

    return segments, labels, input_shape

"""
Create segments and labels for "leave one participant out" experiment on classifying depressed vs non-depressed participants

- dataset_dir:      location of dataset files
- segment_length:   length of each segment
- step:             how many indexes to jump after each iteration
- leave_out_id:     which participant that is left out, a random is selected if left as None

Returns:
- segments
- labels
- left_out_segments
- left_out_correct
- input_shape

"""
def create_segments_and_labels_loo(dataset_dir, segment_length, step, leave_out_id=None):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['afftype'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    left_out_segments = []
    left_out_correct = []

    if leave_out_id == None:
        leave_out_id = random.randint(0, len(scores['number']))

    for i, person in enumerate(scores['number']):
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        if i == leave_out_id:
            if p['afftype'].values[0] == 0:
                left_out_correct = 0
            else:
                left_out_correct = 1

        for j in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[j : j + segment_length]
            
            if i == leave_out_id:
                left_out_segments.append([segment])
            else:
                segments.append([segment])
                
                if p['afftype'].values[0] == 0: 
                    labels.append(0)
                else:
                    labels.append(1)

    labels = np.asarray(labels).astype('float32')
    labels = to_categorical(labels, 2)

    segments = np.asarray(segments).reshape(-1, segment_length, 1)
    left_out_segments = np.asarray(left_out_segments).reshape(-1, segment_length, 1)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    left_out_segments = left_out_segments.reshape(left_out_segments.shape[0], input_shape).astype('float32')
    
    return segments, labels, left_out_segments, left_out_correct, input_shape


"""
Create segments and labels for "leave one participant out" experiment on classifying depression classes

- dataset_dir:      location of dataset files
- segment_length:   length of each segment
- step:             how many indexes to jump after each iteration
- leave_out_id:     which participant that is left out, a random is selected if left as None

Returns:
- segments
- labels
- left_out_segments
- left_out_correct
- input_shape

"""
def create_segments_and_labels_madrs_loo(dataset_dir, segment_length, step, leave_out_id=None):
    scores = pd.read_csv(os.path.join(dataset_dir, 'scores.csv'))
    scores['madrs2'].fillna(0, inplace=True)
    
    segments = []
    labels = []

    left_out_segments = []
    left_out_correct = None

    if leave_out_id == None:
        leave_out_id = random.randint(0, len(scores['number']))

    for i, person in enumerate(scores['number']):
        p = scores[scores['number'] == person]
        filepath = os.path.join(dataset_dir, person.split('_')[0], f'{person}.csv')
        df_activity = pd.read_csv(filepath)

        p_label = None

        for j in range(classes):
            if p['madrs2'].values[0] >= MADRS_VALUES[classes - j - 1]:
                p_label = classes - j - 1
                break

        assert p_label != None, f'Could not get target label: {p}'

        # For the participant that we are leaving out
        if i == leave_out_id:
            left_out_correct = p_label

        for j in range(0, len(df_activity) - segment_length, step):
            segment = df_activity['activity'].values[j : j + segment_length]
            
            # For the participant that we are leaving out
            if i == leave_out_id:
                left_out_segments.append([segment])
            else:
                segments.append([segment])
                labels.append(p_label)

    assert left_out_correct != None and len(left_out_segments) > 0, f'Did not set data for left out participant {leave_out_id}'
    
    labels = np.asarray(labels).astype('float32')
    labels = to_categorical(labels, 3)
    
    segments = np.asarray(segments).reshape(-1, segment_length, 1)
    left_out_segments = np.asarray(left_out_segments).reshape(-1, segment_length, 1)

    num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
    input_shape = num_time_periods * num_sensors

    segments = segments.reshape(segments.shape[0], input_shape).astype('float32')
    left_out_segments = left_out_segments.reshape(left_out_segments.shape[0], input_shape).astype('float32')
    
    return segments, labels, left_out_segments, left_out_correct, input_shape



