from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate, sig) = wav.read("output.wav")
mfcc_feat = mfcc(sig, rate, nfft=610)
fbank_feat = logfbank(sig, rate, nfft=610)

print(fbank_feat[1:3, :])
print()
