#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
import math
import timeit

# Predetermined things, (as the signal has been partially processed)
Fs = 20
F_HIGH = 2
F_LOW  = .025
F_HIGH_ROLLOFF = F_HIGH/4. + F_HIGH
F_LOW_ROLLOFF  = F_LOW - F_LOW/4. 
disp = np.load('data.npy')


def return_psds(data,Fs):
	NFFT_nowindow=len(data)
	f_nowindow, S_nowindow = signal.welch(data, fs=Fs, nperseg=NFFT_nowindow, detrend=None)#, scaling='density')
	NFFT_window=int(len(data)/10)
	f_window, S_window = signal.welch(data, fs=Fs, nperseg=NFFT_window, noverlap=NFFT_window/2, 
	    window='hanning', detrend=None)#, scaling='density')
	return f_nowindow, S_nowindow, f_window, S_window




def freq_time_norm(data, F_LOW, F_HIGH):
	npts = len(data)

	# Calculate the size of the frequency array over which we will normalize, setting the
	# frequency interval to 1/4 the lowest frequency.
	f_increment = F_LOW / 4.
	nf = int( round( (F_HIGH - F_LOW) / f_increment ) + 1 )

	# We will be making a 2nd order, bandpass, butterworth filter
	Filter_Order = 2 	
	f_nyquist = Fs / 2.

	# Initialize filter parameters (a 2nd order bp, has 5 coefficients)
	B = np.zeros((nf,5))
	A = np.zeros((nf,5))

	# We normalize over discrete frequency bands, creating a normalization for each,
	# in terms of normalize frequency.
	for K in range(0, nf):
		f_one = F_LOW + K * f_increment
		f_one_norm = f_one / f_nyquist
		f_two = F_LOW + (K + 1) * f_increment
		f_two_norm = f_two / f_nyquist
		B[K][:], A[K][:] = signal.butter(Filter_Order, [f_one_norm, f_two_norm],btype='band')

	# Initialize normalized output.
	normalize_data = np.zeros(npts)
	
	# Now rebuild the signal, frequency band by frequency band.

	hilbert_time=0
	for L in range(0, nf):
		#print(L,'/',nf)
		narrow_filtered = signal.filtfilt( B[L][:], A[L][:], data)
		# ITS THIS HILBERT TRANSFORM THAT TAKES TO LONG!
		hilbert_tic=timeit.default_timer()
		envelope = np.absolute(signal.hilbert(narrow_filtered))
		hilbert_time=hilbert_time+(timeit.default_timer()-hilbert_tic)
		envelope_norm = narrow_filtered / envelope
		normalize_data = normalize_data + envelope_norm
	
	print('Total time spent Hilberting=',hilbert_time)

	normalize_data = normalize_data / np.sqrt(nf)
	return normalize_data





npts=len(disp)
time=np.linspace(0,npts/Fs,npts)

# Plot the signal
plt.figure(1)
plt.plot(time, disp)
plt.xlabel('time (s)')
plt.ylabel('y')
plt.xlim(xmin=time[0])
plt.xlim(xmax=time[-1])
plt.title('Non-whitened Signal')
plt.show(block=False)


# Plot the PSD
f_nowindow, S_nowindow, f_window, S_window = return_psds(disp,Fs)
plt.figure(2)
plt.plot(f_nowindow, 10*np.log10(S_nowindow))
plt.plot(f_window, 10*np.log10(S_window))
plt.xlabel('f (Hz)')
plt.ylabel('PSD')
plt.xlim(xmax=F_HIGH_ROLLOFF)
plt.xlim(xmin=F_LOW_ROLLOFF)
plt.title('Non-whitened PSD')
plt.show(block=False)

# Plot the whitened signal
disp_FTN=freq_time_norm(disp, F_LOW, F_HIGH)
plt.figure(3)
plt.plot(time, disp_FTN)
plt.xlabel('time (s)')
plt.ylabel('y')
plt.xlim(xmin=time[0])
plt.xlim(xmax=time[-1])
plt.title('Whitened Signal')
plt.show(block=False)

# Plot the whitened PSD
f_nowindow_FTN, S_nowindow_FTN, f_window_FTN, S_window_FTN = return_psds(disp_FTN,Fs)
plt.figure(4)
plt.plot(f_nowindow_FTN, 10*np.log10(S_nowindow_FTN))
plt.plot(f_window_FTN, 10*np.log10(S_window_FTN))
plt.xlabel('f (Hz)')
plt.ylabel('PSD')
plt.xlim(xmax=F_HIGH_ROLLOFF)
plt.xlim(xmin=F_LOW_ROLLOFF)
plt.title('Whitened PSD')
plt.show(block=False)
