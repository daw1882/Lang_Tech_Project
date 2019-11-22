# Keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# Rest
import os
import keras
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

input_duration = 3

dir_list = os.listdir("training_set/")
dir_list.sort()

dir_list2 = os.listdir("training_set2/")
dir_list2.sort()

data_df = pd.DataFrame(columns=['path', 'age', 'emotion'])
count = 0

# Gets the TESS training set
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

# Gets the RAVDESS training set
for i in dir_list2:
    file_list = os.listdir('training_set2/' + i)
    for f in file_list:
        nm = f.split('.')[0].split('-')
        path = 'training_set2/' + i + '/' + f
        age = 'unknown'
        if nm[2] == '05':
            emotion = 0
        elif nm[2] == '07':
            emotion = 1
        elif nm[2] == '06':
            emotion = 2
        elif nm[2] == '03':
            emotion = 3
        elif nm[2] == '01' or nm[2] == '02':
            emotion = 4
        elif nm[2] == '08':
            emotion = 5
        elif nm[2] == '04':
            emotion = 6
        else:
            emotion = -1
        data_df.loc[count] = [path, age, emotion]
        count += 1

# Labels all the data
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
a = data_df.label.value_counts()

# Getting features
data = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data_df))):
    X, sample_rate = librosa.load(data_df.path[i], res_type='kaiser_fast', duration=input_duration,
                                  sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[i] = [feature]

df = pd.DataFrame(data['feature'].values.tolist())
labels = data_df.label

newdf = pd.concat([df, labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
rnewdf = rnewdf.fillna(0)

# Splitting into test data and training data randomly
X = rnewdf.drop(['label'], axis=1)
y = rnewdf.label
xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
for train_index, test_index in xxx.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
# Encodes the labels numerically to be understood by the model
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)


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


# New model
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1], 1), data_format='channels_first'))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(7))
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

# Plotting Model Summary
model.summary()

# Compile your model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', fscore])

# Model Training
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)

# Change the model name accordingly.
mcp_save = ModelCheckpoint('model/RAVDESS_model.h5', save_best_only=True, monitor='val_loss', mode='min')
cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=700,
                       validation_data=(x_testcnn, y_test),
                       callbacks=[mcp_save, lr_reduce])


# Plotting the Train Valid Loss Graph
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Saving the combined.json
model_json = model.to_json()
with open("model/RAVDESS_model.json", "w") as json_file:
    json_file.write(model_json)

# loading json and creating model
json_file = open('model/RAVDESS_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/RAVDESS_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

data_test = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data_df))):
    X, sample_rate = librosa.load(data_df.path[i], res_type='kaiser_fast', duration=input_duration, sr=22050 * 2,
                                  offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data_test.loc[i] = [feature]

test_valid = pd.DataFrame(data_test['feature'].values.tolist())
test_valid = test_valid.fillna(0)
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
actual=test_valid_lb.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})
finaldf = actualdf.join(preddf)

print(finaldf.groupby('actualvalues').count())
print(finaldf.groupby('predictedvalues').count())


