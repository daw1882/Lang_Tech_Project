import IPython.display as ipd
import librosa.display
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json


def get_emotions(filename):
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']
    # loading json and creating model
    json_file = open('model/combined.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model/best_model.h5")
    print("Loaded model from disk")

    print(filename)

    samples, sample_rate = librosa.load(filename)

    # Trim the silence voice
    aa, bb = librosa.effects.trim(samples, top_db=30)

    # Silence trimmed Sound by librosa.effects.trim()
    ipd.Audio(aa, rate=sample_rate)

    data = pd.DataFrame(columns=['feature'])
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast',
                                  duration=3, sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[0] = [feature[:215]]

    df = pd.DataFrame(data['feature'].values.tolist())
    df = df.fillna(0)

    test = np.zeros((1, 215))
    test[:df.shape[0], :df.shape[1]] = np.array(df)
    test = np.expand_dims(test, axis=2)

    lb = LabelEncoder()
    test_valid_lb = np.array(labels)
    lb.fit_transform(test_valid_lb)

    preds = loaded_model.predict(test, batch_size=16, verbose=1)
    total = 0
    for pred in preds[0]:
        total += pred
    preds1 = preds.argmax(axis=1)[0]
    percentages = []
    for i in range(7):
        i = np.array(i)
        i = i.astype(int).flatten()
        print(lb.inverse_transform(i)[0], ": %.2f%%" % (preds[0][i]/total * 100))
        percentages.append(preds[0][i][0]/total * 100)
    abc = preds1.astype(int).flatten()
    predictions = (lb.inverse_transform(abc))
    print()
    print('Detected:', predictions[0])
    return percentages, predictions[0]

