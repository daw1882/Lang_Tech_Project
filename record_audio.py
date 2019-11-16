import pyaudio
import wave
import keyboard
import time

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 12200  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Will start recording when SPACE is pressed.')

keyboard.wait('space')
running = True

time.sleep(0.5)
print("Recording. Press SPACE to stop.")

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
while running:
    data = stream.read(chunk)
    frames.append(data)
    if keyboard.is_pressed('space'):
        running = False

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording.')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

from keras.models import load_model
import numpy as np
import librosa
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")


def predict(audio):
    prob = loaded_model.predict(audio)
    index = np.argmax(prob[0])
    return index


X, sample_rate = librosa.load('output.wav', res_type='kaiser_fast', duration=3,
                              sr=22050, offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
feature = mfccs

predict(mfccs)



