import os
import random
import sys

## Package
import glob
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
import json

## Keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K

## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
#py.init_notebook_mode(connected=True)

input_duration = 3

dir_list = os.listdir("training_set/")
dir_list.sort()
print(dir_list)

data_df = pd.DataFrame(columns=['path', 'age', 'emotion'])
count = 0
for i in dir_list:
    file_list = os.listdir('training_set/' + i)
    for f in file_list:
        nm = f.split('.')[0].split('_')
        path = 'training_set/' + i + '/' + f
        age = i.split('_')[0]
        if nm[2] == 'angry':
            emotion = 0
        elif nm[2] == 'disgust':
            emotion = 1
        elif nm[2] == 'fear':
            emotion = 2
        elif nm[2] == 'happy':
            emotion = 3
        elif nm[2] == 'neutral':
            emotion = 4
        elif nm[2] == 'ps':
            emotion = 5
        elif nm[2] == 'sad':
            emotion = 6
        else:
            emotion = -1
        data_df.loc[count] = [path, age, emotion]
        count += 1

#print(len(data_df))
#print(data_df.emotion[1000])

filename = data_df.path[1021]
#print(filename)

samples, sample_rate = librosa.load(filename)
print(len(samples), sample_rate)


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# freqs, times, spectrogram = log_specgram(samples, sample_rate)

# For waveform and spectrogram
# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(211)
# ax1.set_title('Raw wave of ' + filename)
# ax1.set_ylabel('Amplitude')
# librosa.display.waveplot(samples, sr=sample_rate)
#
# ax2 = fig.add_subplot(212)
# ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
#            extent=[times.min(), times.max(), freqs.min(), freqs.max()])
# ax2.set_yticks(freqs[::16])
# ax2.set_xticks(times[::16])
# ax2.set_title('Spectrogram of ' + filename)
# ax2.set_ylabel('Freqs in Hz')
# ax2.set_xlabel('Seconds')
#plt.show()

# For mel power spectrogram
# mean = np.mean(spectrogram, axis=0)
# std = np.std(spectrogram, axis=0)
# spectrogram = (spectrogram - mean)/std
# aa, bb = librosa.effects.trim(samples, top_db=30)
#
# S = librosa.feature.melspectrogram(aa, sr=sample_rate, n_mels=128)
# log_S = librosa.power_to_db(S, ref=np.max)
# plt.figure(figsize=(12, 4))
# librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
# plt.title('Mel power spectrogram ')
# plt.colorbar(format='%+02.0f dB')
# plt.tight_layout()
#plt.show()

# For MFCC
# mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
# delta2_mfcc = librosa.feature.delta(mfcc, order=2)
# plt.figure(figsize=(12, 4))
# librosa.display.specshow(delta2_mfcc)
# plt.ylabel('MFCC coeffs')
# plt.xlabel('Time')
# plt.title('MFCC')
# plt.colorbar()
# plt.tight_layout()
#plt.show()

# ipd.Audio(samples, rate=sample_rate)
# ipd.Audio(aa, rate=sample_rate)
# sample_cut = samples[10000:-12500]
# ipd.Audio(sample_cut, rate=sample_rate)

label_list = []
for i in range(len(data_df)):
    if data_df.emotion[i] == 0:
        lb = 'angry'
    elif data_df.emotion[i] == 1:
        lb = 'disgust'
    elif data_df.emotion[i] == 2:
        lb = 'fear'
    elif data_df.emotion[i] == 3:
        lb = 'happy'
    elif data_df.emotion[i] == 4:
        lb = 'neutral'
    elif data_df.emotion[i] == 5:
        lb = 'surprised'
    elif data_df.emotion[i] == 6:
        lb = 'sad'
    else:
        lb = 'none'
    label_list.append(lb)

data_df['label'] = label_list
# print(data_df.head())
# print()
# print(data_df.label.value_counts().keys())

# Plotting the emotion distribution


def plot_emotion_dist(dist, color_code='#C2185B', title="Plot"):
    """
    To plot the data distributioin by class.
    Arg:
      dist: pandas series of label count.
    """
    tmp_df = pd.DataFrame()
    tmp_df['Emotion'] = list(dist.keys())
    tmp_df['Count'] = list(dist)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax = sns.barplot(x="Emotion", y='Count', color=color_code, data=tmp_df)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()


a = data_df.label.value_counts()
# plot_emotion_dist(a, "#2962FF", "Emotion Distribution")

# Getting features
data = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data_df))):
    X, sample_rate = librosa.load(data_df.path[i], res_type='kaiser_fast', duration=input_duration,
                                  sr=22050*2, offset=0.5)
#     X = X[10000:90000]
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[i] = [feature]

# print(data.head())

df = pd.DataFrame(data['feature'].values.tolist())
labels = data_df.label
# print(df.head())

newdf = pd.concat([df, labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
# print(len(rnewdf))
# print()
# print(rnewdf.head(10))
# print()
# print(rnewdf.isnull().sum().sum())
# print()
rnewdf = rnewdf.fillna(0)
# print(rnewdf.head(10))


def plot_time_series(data1):
    """
    Plot the Audio Frequency.
    """
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data1)), data1)
    plt.show()


def noise(data1):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005 * np.random.uniform() * np.amax(data1)
    data1 = data1.astype('float64') + noise_amp * np.random.normal(size=data1.shape[0])
    return data1


def shift(data1):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high=5) * 500)
    return np.roll(data1, s_range)


def stretch(data1, rate=0.8):
    """
    Streching the Sound.
    """
    data1 = librosa.effects.time_stretch(data1, rate)
    return data


def pitch(data1, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data1 = librosa.effects.pitch_shift(data1.astype('float64'),
                                       sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data


def dyn_change(data1):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=1.5, high=3)
    return (data1 * dyn_change)


def speedNpitch(data1):
    """
    peed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(data1), speed_fac), np.arange(0, len(data1)), data1)
    minlen = min(data1.shape[0], tmp.shape[0])
    data1 *= 0
    data1[0:minlen] = tmp[0:minlen]
    return data1


# X, sample_rate = librosa.load(data_df.path[216], res_type='kaiser_fast', duration=4,
#                               sr=22050*2, offset=0.5)
# plot_time_series(X)
# ipd.Audio(X, rate=sample_rate)
# plt.show()

X = rnewdf.drop(['label'], axis=1)
print(X.head())
#X = X.drop(['path'], axis=1)
#X = X.drop(['age'], axis=1)
y = rnewdf.label
xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
for train_index, test_index in xxx.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# print(y_train.value_counts())
# print(y_test.value_counts())
# print(X_train.isna().sum().sum())

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# print()
#print('Y-train', y_train.shape)
# print('==============================================')
# print()
#print('X-train', X_train.shape)
# print()
# print(y_test)
#print('y-test', y_test.shape)

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

#print(x_traincnn.shape)
#print(x_testcnn.shape)


# Set up Keras util functions
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@tf.function
def fscore(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return float(0)

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


# # New model
# model = Sequential()
# model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1], 1), data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(Conv1D(256, 8, padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(MaxPooling1D(pool_size=(8)))
# model.add(Conv1D(128, 8, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 8, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 8, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 8, padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(MaxPooling1D(pool_size=(8)))
# model.add(Conv1D(64, 8, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(64, 8, padding='same'))
# model.add(Activation('relu'))
# model.add(Flatten())
# # Edit according to target class no.
# model.add(Dense(7))
# model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
#
# # Plotting Model Summary
# model.summary()
#
# # Compile your model
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', fscore])
#
# # Model Training
# lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
# # Please change the model name accordingly.
# mcp_save = ModelCheckpoint('model/model.h5', save_best_only=True, monitor='val_loss', mode='min')
# cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=700,
#                        validation_data=(x_testcnn, y_test),
#                        callbacks=[mcp_save, lr_reduce])
#
#
# # Plotting the Train Valid Loss Graph
# plt.plot(cnnhistory.history['loss'])
# plt.plot(cnnhistory.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# # Saving the model.json
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# loading json and creating model
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

data_test = pd.DataFrame(columns=['feature'])
print(data_df[1000:1010])
for i in tqdm(range(len(data_df))):
    X, sample_rate = librosa.load(data_df.path[i], res_type='kaiser_fast', duration=input_duration, sr=22050 * 2,
                                  offset=0.5)
    #     X = X[10000:90000]
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data_test.loc[i] = [feature]

test_valid = pd.DataFrame(data_test['feature'].values.tolist())
test_valid = test_valid.fillna(0)
print(test_valid.head())
test_valid = np.array(test_valid)
test_valid_lb = np.array(data_df.label)
lb = LabelEncoder()
test_valid_lb = np_utils.to_categorical(lb.fit_transform(test_valid_lb))
test_valid = np.expand_dims(test_valid, axis=2)


preds = loaded_model.predict(test_valid,
                         batch_size=16,
                         verbose=1)
preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})
print(preddf[1000:1010])
actual=test_valid_lb.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})
print(actualdf[1000:1010])
finaldf = actualdf.join(preddf)
print(finaldf[1000:])

print(finaldf.groupby('actualvalues').count())
print(finaldf.groupby('predictedvalues').count())


