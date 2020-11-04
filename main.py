
import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_gdf
from mne.decoding import CSP

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


'''
Parameters:
'''
tmin = 0
tmax = 10


'''
Parameters:
'''
tmin = 0
tmax = 10

file_P01_Run3 = read_raw_gdf('..\P01\P01 Run 3.gdf', stim_channel=None, eog=[61,62,63])
eventDescription_offline_paradigm = {
    '768': "trial start",
    '785': "beep",
    '786': "fixation cross",
    '776': "supinationclass cue",
    '777': "pronationclass cue",
    '779': "hand openclass cue",
    '925': "palmar graspclass cue",
    '926': "ateral graspclass cue",
}
# narrow it down to two
eventDescription_offline_paradigm = {
    '785': "beep",
    '925': "palmar graspclass cue",
}

# events = [number of events, position, event code, channel, duration]
# file_P01_Run3._raw_extras[0]['events']
event, _ = mne.events_from_annotations(file_P01_Run3)


# build event id and filter 1-7 id
event_id = {}
legal_event_list = []
for i in _: #handle event_id
    if i not in eventDescription_offline_paradigm:
        continue
    legal_event_list.append(_[i])
    event_id[eventDescription_offline_paradigm[i]] = _[i]


print('event id')
print(event_id)
epochs = mne.Epochs(file_P01_Run3, event, event_id, tmin=-0., tmax=1, baseline=None, event_repeated = 'merge', preload=True)


labels = epochs.events[:,-1]
print(labels)

# format: trials, channels, samples
X = epochs.get_data() * 1000
y = labels

kernels, chans, samples = 1, 64, 257


# Spliting dataset
X_train = X[0:40,]
Y_train = y[0:40]
X_validate = X[40:60,]
Y_validate = y[40:60]
X_test = X[60:,]
Y_test = y[60:]

print(Y_train)


# Convert label to one-hot encodings
Y_train = np_utils.to_categorical(Y_train - 1)
Y_validate = np_utils.to_categorical(Y_validate - 1)
Y_test = np_utils.to_categorical(Y_test - 1)

#convert to (trials, kernels, channels, samples) format.
# contains 64 channels and 257 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
X_validate = X_validate.reshape(X_validate.shape[0], kernels, chans, samples)
X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 15, Chans = chans, Samples = samples,
               dropoutRate = 0.5, kernLength = 128, F1 = 8, D = 2, F2 = 16)
model.summary()


# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

# count number of parameters in the model
numParams    = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='./checkpoint.h5', verbose=1,
                               save_best_only=True)


# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
# unused
class_weights = {0:1, 1:1, 2:1, 3:1}
with tf.device('/device:GPU:0'):
    fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 150,
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])


# load optimal weights
model.load_weights('./checkpoint.h5')

###############################################################################
# make prediction on test set.
###############################################################################

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))
