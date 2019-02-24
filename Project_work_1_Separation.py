# coding: utf-8

# Course: SGN-14007 Introduction to Audio Processing
# Project work: Implementing an audio signal processing algorithm. Algorithm based on paper: 
# 				Separation of a monaural audio signal into harmonic/percussive components by complementary diffusion on spectrogram.
# 				Topic 1: Separation of drums from music signals.
#				Petri Tikka - 206059, Joonas Linnosmaa -  

# import Python modules
from matplotlib.pyplot import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy.constants import pi
from scipy.signal import stft, istft, lfilter
from scipy.signal import freqz, convolve
from numpy.fft.fftpack import fft
import librosa
import librosa.display
import math

# load the sample
fs, f_t = wavfile.read('rhythm_birdland.wav')
total_samples = len(f_t)
duration = total_samples/float(fs)

# normalize x so that its value is between [-1.00, 1.00]
f_t = f_t.astype('float64') / float(np.max(np.abs(f_t)))

# set a window of duration 20 ms
win_duration = 20/1000.0
print 'WIN_DURATION: ' + str(win_duration)
print 'FS: ' + str(fs) 
win_length = int(fs*win_duration)
print 'WIN_LEN: ' + str(win_length)

# set an overlap ratio of 50 %
hop_length = int(win_length/2)

# 1. Calculate F_h_i, the STFT of an input signal f(t).
#    Compute spectrogram
F_h_i = librosa.stft(f_t, n_fft=win_length, hop_length=hop_length)

D = librosa.amplitude_to_db(np.abs(F_h_i), ref=np.max)
librosa.display.specshow(D, sr=fs, y_axis='linear', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of rhythm_birdland.wav')
plt.show()

# 2. Calculate a range-compressed version of the power spectrogram by
#    W_h_i = |F_h_i|^2*gamma, (0 <= gamma <= 1).
gamma = 0.3
W_h_i = np.abs(F_h_i)**(2*gamma)

# 3. Set initial values as:
H = 0.5*W_h_i 
P = 0.5*W_h_i 
# for all h and i and set k = 0

# 4. Calculate the update variable delta_k defined as:
#    delta_k = alfa*((H_h_i-1^k - 2*H_h_i^k + H_h_i+1^k) / 4) - (1 - alfa) * (P_h-1_i^k - 2*P_h_i^k + P_h+1_i^k) / 4)
#    The balance parameter alfa (0 < alfa < 1) controls the strength of the diffusion along the vertical and the horizontal directions.
alfa = 0.3
k = 0
k_max = 50
while k < k_max:
	# We have to padding in order to have columns that refer to index-1 and index +1:
	H_h_i = np.pad(H, ((0,0),(1,1)), 'constant')
	P_h_i = np.pad(P, ((1,1),(0,0)), 'constant')
	var_1 = ((H_h_i[:, 0:-2] - 2*H_h_i[:, 1:-1] + H_h_i[:, 2:]) / 4)
	var_2 = (1 - alfa) * ((P_h_i[0:-2, :] - 2*P_h_i[1:-1, :] + P_h_i[2:, :]) / 4)
	delta_k = alfa * var_1 - (1 - alfa) * var_2

	# 5. Update H_h_i and P_h_i as:
	zeros = np.zeros(H.shape)
	# Update variable is divided according harmonic and percussive
	H = np.minimum(np.maximum(H + delta_k, zeros), W_h_i)
	P = W_h_i - H
	# 6. Increment k. If k < kmax -1 (kmax: the maximum number
	#    of iterations), then, go to step 4, else, go to step 7.
	k = k + 1

# 7. Binarize the separation result as:
#	 We take the whole W and turn to zero all elements where H >= P 
P_binary = W_h_i.copy()
H_binary = W_h_i.copy()
P_binary[H >= P] = 0
H_binary[H < P] = 0

# 8. Convert H and P into waveforms by:
H_binary = np.multiply(H_binary**(0.5/gamma),np.exp(1j*np.angle(F_h_i)))
P_binary = np.multiply(P_binary**(0.5/gamma),np.exp(1j*np.angle(F_h_i)))

h_t = librosa.istft(H_binary,win_length=win_length, hop_length=hop_length, length = total_samples)
p_t = librosa.istft(P_binary,win_length=win_length, hop_length=hop_length, length = total_samples)

# plot spectrograms
HT = librosa.amplitude_to_db(np.abs(librosa.stft(h_t, n_fft = win_length, hop_length = hop_length)), ref=np.max)
librosa.display.specshow(HT, sr=fs, y_axis='linear', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Harmonic spectrogram of rhythm_birdland.wav')
plt.show()

PT = librosa.amplitude_to_db(np.abs(librosa.stft(p_t, n_fft = win_length, hop_length = hop_length)), ref=np.max)
librosa.display.specshow(PT, sr=fs, y_axis='linear', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Percussive spectrogram of rhythm_birdland.wav')
plt.show()

# Signal to Noise Ratio:
# Calculated as instructed during exercises.
SNR_harmonic = 10*np.log10(np.sum(f_t**2)/np.sum(h_t**2))
SNR_percussive = 10*np.log10(np.sum(f_t**2)/np.sum(p_t**2))
print 'Signal-to-noise ratio of original signal and harmonic: ' + str(SNR_harmonic)
print 'Signal-to-noise ratio of original signal and percussive: ' + str(SNR_percussive)

# If s(t) contains the monophonic mixture of the instruments and y(t) is the mixture of your
# separated harmonic components yh(t) and percussive yp(t) components, then e(t) is the difference
# between s(t) and y(t), i.e., e(t)=s(t)-y(t). Ideally e(t) should be close to zero all the time, 
# and SNR very high, (+20 dB or better). Below there's an example sample of the separated harmonic and percussive components.
y_t = h_t + p_t
e_t = f_t - y_t
SNR = 10*np.log10(np.sum(f_t**2)/np.sum(e_t**2))
print 'Signal-to-noise ratio of original signal and mixture of separated harmonic and percussive components: ' + str(SNR)

# Translate to wavfiles:
audio_name = 'Harmonic_signal_rhythm_birdland.wav'
wavfile.write(audio_name, fs, h_t)
	 
audio_name = 'Percussive_signal_rhythm_birdland.wav'
wavfile.write(audio_name, fs, p_t)	 