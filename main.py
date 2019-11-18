from emotion_analyzer import *
from record_audio import *

if __name__ == '__main__':
    record_and_save()
    get_emotions('mchocola.wav')
