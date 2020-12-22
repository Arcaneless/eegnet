import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import mne
from mne import Epochs, pick_types, find_events
from mne.filter import filter_data
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_gdf
from mne.decoding import CSP
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib tk


# =========================================
#               PARAMETERS
# =========================================
tmin = -0.1
tmax = 2

# Reading segment
object_names = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10']
runMove = [3, 4, 5, 6, 7, 10, 11, 12, 13] # these are runs with attempted movement

# Paradigm description
# this two are the events we will be focus on
eventDescription_offline_paradigm = {
   '779': "hand openclass cue",
   '925': "palmar graspclass cue",
}

# =========================================
#               VARIABLES
# =========================================
files = []
# The target epochs
result_epochs = None
event_id = {}  # ids

# Datas
X = []
y = []
X_train = []
X_test = []
Y_train = []
Y_test = []
kernels, chans, samples = 0,0,0

# Model
model = None
losses = []
accs = []


# =========================================
#               CALLBACKS
# =========================================


checkpointer = ModelCheckpoint(filepath='./checkpoint.h5', verbose=1,
                                   save_best_only=True)


class LossAndAccRecord(Callback):
    def on_batch_end(self, batch, logs=None):
        losses.append(logs["loss"])
        accs.append(logs["accuracy"])

# =========================================
#               FUNCTIONS
# =========================================

# from object name


def read_file(object_name):
    for i in runMove:
        files.append(read_raw_gdf(f'..\\{object_name}\\{object_name} Run {i}.gdf', stim_channel=None, eog=[61, 62, 63], preload=True))


def extract_epochs():
    global result_epochs, event_id
    epochs_list = []
    for file in files:
        # band pass filter
        # file.filter(0.1, 100, method='fir')
        event, _ = mne.events_from_annotations(file)
        # build event id and filter 1-7 id

        for i in _:  # handle event_id
            if i not in eventDescription_offline_paradigm:
                continue
            event_id[eventDescription_offline_paradigm[i]] = _[i]

        print(f'event id: {event_id}')
        epochs = mne.Epochs(file, event, event_id, tmin=tmin, tmax=tmax, baseline=None, event_repeated='merge', preload=True)
        epochs_list.append(epochs)

    result_epochs = mne.concatenate_epochs(epochs_list)


def plot_psd(num):
    assert 0 <= num < len(files)
    # plotting
    files[num].plot_psd(fmax=30)


def build_data():
    global X, y, X_train, X_test, Y_train, Y_test, kernels, chans, samples
    # 0: not supinationclass or pronationclass cue
    labels = result_epochs.events[:, -1]
    # labels = np.array(list(map(lambda x:  0 if (not x == 9) else x, labels)))
    print(labels)

    # format: trials, channels, samples
    X = result_epochs.get_data()
    y = labels

    kernels, chans, samples = 1, X.shape[1], X.shape[2]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def encode_reshape():
    global X, y, X_train, X_test, Y_train, Y_test
    encoder = OneHotEncoder()
    encoder.fit([[x] for x in Y_train])
    Y_train = encoder.transform([[x] for x in Y_train]).toarray()
    Y_test = encoder.transform([[x] for x in Y_test]).toarray()

    # convert to (trials, kernels, channels, samples) format.
    # contains 'chans' channels and 'samples' time-points. Set the number of kernels to 'kernels'.
    X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
    X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


def init_model():
    global model
    classes_num = len(eventDescription_offline_paradigm)
    # configure the EEGNet-8,2,16 model with kernel length of 257 samples (other
    # model configurations may do better, but this is a good starting point)
    # class num is one
    model = EEGNet(nb_classes=classes_num, Chans=chans, Samples=samples,
                   dropoutRate=0.8, kernLength=128, F1=8, D=2, F2=16)
    model.summary()


def compile_model():
    global model, losses, accs
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])


def run_model():
    global model
    model.fit(X_train, Y_train, batch_size=20, epochs=50,
                            verbose=1, callbacks=[checkpointer, LossAndAccRecord()])


def predict_model():
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))

    # RESULTS
    print("Classification accuracy: %.4f " % (acc))
    print(confusion_matrix(Y_test.argmax(axis=-1), preds))

    # fig, ax = plt.subplots(2)
    # ax[0].plot(losses)
    # ax[0].set_ylabel('Loss')
    # ax[0].set_xlabel('Epochs')
    # ax[1].plot(accs)
    # ax[1].set_ylabel('Accuracy')
    # ax[1].set_xlabel('Epochs')
    # plt.show()

    return acc

# =========================================
#               MAIN
# =========================================


if __name__ == '__main__':
    total_arr = {
        'Object Name': [],
        'Mean': [],
        'Std': []
    }
    for object_name in object_names:
        read_file(object_name)
        # for i in range(len(runMove)):
        #     plot_psd(i)
        extract_epochs()
        build_data()
        encode_reshape()
        acc_arr = np.array([])
        trials = 10
        for i in range(trials):
            print(f'{object_name}: Run {i+1}')
            init_model()
            compile_model()
            run_model()
            acc = predict_model()
            acc_arr = np.append(acc_arr, [acc])

        print(f'average accuracy over {trials} trials: {np.mean(acc_arr)}')
        total_arr['Object Name'].append(object_name)
        total_arr['Mean'].append(np.mean(acc_arr))
        total_arr['Std'].append(np.std(acc_arr))

    print(pd.DataFrame(data=total_arr))