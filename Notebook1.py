#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:43:25 2021

This file .py is supposed to contain all the functionalities/classes provided
in the first notebook of the Environmental Sound Classification project that 
can be useful also for the other notebooks.

Project folder: https://github.com/MattiaPujatti/Environmental-Sound-Classification

@author: mattia
"""

import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from tqdm.notebook import tqdm
import shutil

def get_label_map():
    """ Since the labels are represented by a numerical class in [0-49], it would 
    be nice to have a way to associate the name of the class to the label."""
    label_map = {}
    label_df = pd.read_csv(os.getcwd() + '/ESC-50-master/ESC-50-master/meta/esc50.csv')
    for i in range(50):
        label_map[i] = label_df.loc[label_df["target"]==i, "category"].unique()[0]
    return label_map

def macro_categories_map(label):
    """ And the same approach can be useful to distinguish among different macro-categories.
    Each macro-category corresponds to a range of "target" parameters among the labels."""
    if label in [i for i in range(10)]   : return 'Animals' 
    if label in [i for i in range(10,20)]: return 'Natural soundscapes and water sounds'
    if label in [i for i in range(20,30)]: return 'Human, non-speech sounds'  
    if label in [i for i in range(30,40)]: return 'Interior/domestic sounds'
    if label in [i for i in range(40,50)]: return 'Exterior/urban noises'   
    else:
        print("Invalid label input")
        return None



class Clip():
    """A class to handle those 5.0 seconds clips and extract features from them."""
    
    class Audio:
        """ The actual audio data of the clip.
            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage, keeping just the 
            features computed but not the entire sound vector.
        """
        
        def __init__(self, path):
            self.path = path
            self.timelength = 5.0
        
        def __enter__(self):
            # In order to prevent having clips longer or smaller than 5 seconds
            self.data, self.rate = librosa.load(self.path, sr=None, duration=self.timelength)
            rawaudiosize = self.timelength*self.rate
            if self.data.shape[0] < rawaudiosize:
                self.raw = np.pad(self.data, pad_width=(0,int(rawaudiosize-len(self.data))), 
                                   mode='constant', constant_values=0.)
            else:
                self.raw = self.data
                
            return self
        
        def __exit__(self, exception_type, exception_value, traceback):
            if exception_type is not None:
                print(exception_type, exception_value, traceback)
            del self.data
            del self.raw
            
            
            
    def __init__(self, path, label_map):
        self.clipname = os.path.basename(path)
        self.path = path
        
        self.fold = int(self.clipname.split('-')[0])
        self.clipID = int(self.clipname.split('-')[1])
        self.take = self.clipname.split('-')[2]
        self.target = int(self.clipname.split('-')[3].split('.')[0])
        self.category = label_map[self.target]
        
        self.audio = Clip.Audio(self.path)
        

    def Compute_Features(self, features_list=[]):
        """ Exploit the functions provided by the librosa library to compute several audio analysis
        features. Available:
        * spectral centroid, spectral rolloff, spectral bandwidth
        * zero-crossing rate
        * mfcc
        * chromagram
        * delta, delta-delta
        * log-energy, delta-log-energy, delta-delta-log-energy
        If the input values is "all" -> compute all the features listed above
        """
        if features_list == "all": 
            features_list = ["spectral centroid", "spectral rolloff", "spectral bandwidth", 
                             "zero-crossing rate", "mfcc", "chromagram", "delta", "delta-delta", 
                             "energy", "delta-energy", "delta-delta-energy"]
        self.features = {}
        
        with self.audio as audio:
            if "spectral centroid" in features_list:
                self._compute_spectral_centroid(audio.raw, audio.rate)
            if "spectral rolloff" in features_list:
                self._compute_spectral_rolloff(audio.raw, audio.rate)
            if "spectral bandwidth" in features_list:
                self._compute_spectral_bandwidth(audio.raw, audio.rate)
            if "zero-crossing rate" in features_list:
                self._compute_zero_crossing_rate(audio.raw)
            if "mfcc" in features_list:
                self._compute_mfcc(audio.raw, audio.rate)
            if "chromagram" in features_list:
                self._compute_chromagram(audio.raw, audio.rate)
            if "delta" in features_list:
                self._compute_delta(self.audio.raw, audio.rate)
            if "delta-delta" in features_list:
                self._compute_delta_delta(audio.raw, audio.rate)
            if "energy" in features_list:
                self._compute_energy(audio.raw)
            if "delta-energy" in features_list:
                self._compute_delta_energy(audio.raw)
            if "delta-delta-energy" in features_list:
                self._compute_delta_delta_energy(audio.raw)
        return 
    
    def _compute_spectral_centroid(self, audio_track, audio_rate):
        """ The spectral centroid indicates at which frequency the energy of a spectrum is centered 
        upon or, in other words, it indicates where the "center of mass" for a sound is located.
        NFRAMES = (clip_length * frame_rate)/ frame_size 
        Since the frame size = 512 samples by default, for those clips NFRAMES = 431."""
        
        if "spectral centroid" not in self.features: 
            self.features["spectral centroid"] = librosa.feature.spectral_centroid(audio_track, sr=audio_rate)[0]            
        return self.features["spectral centroid"]
    
    def _compute_spectral_rolloff(self, audio_track, audio_rate):
        """The spectral roll off is a measure of the shape of the signal. It represents the frequency 
        at which high frequencies decline to 0. 
        Return a vector with size NFRAMES = 431."""
        
        if "spectral rolloff" not in self.features: 
            self.features["spectral rolloff"] = librosa.feature.spectral_rolloff(audio_track, sr=audio_rate)[0]            
        return self.features["spectral rolloff"]
    
    def _compute_spectral_bandwidth(self, audio_track, audio_rate):
        """The bandwidth is defined as the width of the band of light at one-half the peak maximum 
        (or full width at half maximum [FWHM]) and represents the portion of the spectrum which contains 
        most of the energy of the signal.
        Return a vector of size NFRAMES = 431."""
        
        if "spectral bandwidth" not in self.features: 
            self.features["spectral bandwidth"] = librosa.feature.spectral_bandwidth(audio_track, sr=audio_rate, p=2)[0]            
        return self.features["spectral bandwidth"]
    
    def _compute_zero_crossing_rate(self, audio_track):
        """The zero-crossing rate is the rate at which a signal changes from positive to zero to negative 
        or from negative to zero to positive.
        Return a vector of size NFRAMES = 431."""
        
        if "zero-crossing rate" not in self.features: 
            self.features["zero-crossing rate"] = librosa.feature.zero_crossing_rate(audio_track, pad=False)
        return self.features["zero-crossing rate"]
    
    def _compute_mfcc(self, audio_track, audio_rate):
        """The Mel-Frequency-Cepstral-Coefficient of a signal are a small set of features (usually about 10–20 
        but we will keep) only the first 13 of each frame), which concisely describe the overall shape of a 
        spectral envelope modeling the characteristics of the human voice (according to the Mel scale).
        The first coefficient is usually discarded because it is less interesting.
        The function returns a vector of shape (12, NFRAMES) = (12,431)."""
        
        if "mfcc" not in self.features: 
            self.features["mfcc"] = librosa.feature.mfcc(audio_track, sr=audio_rate, n_mfcc=13)[1:]
        return self.features["mfcc"]
    
    def _compute_chromagram(self, audio_track, audio_rate):
        """A chromagram is typically a 12-element feature vector indicating how much energy of each pitch 
        class, {C, C#, D, D#, E, …, B}, is present in the signal. In short, it provides a way to describe 
        similarities between music pieces).
        The function returns a vector of shape (12, NFRAMES) = (12,431)."""
        
        if "chromagram" not in self.features: 
            self.features["chromagram"] = librosa.feature.chroma_stft(audio_track, sr=audio_rate)
        return self.features["chromagram"]
    
    def _compute_delta(self, audio_track, audio_rate):
        """Delta are a set of coefficients defined as the local derivatives of the mfcc (or logenergy) 
        coefficients. The function returns a vector of size (12, NFRAMES) = (12,431)."""
        
        if "delta" not in self.features: 
            self.features["delta"] = librosa.feature.delta(self._compute_mfcc(audio_track, audio_rate), order=1)
        return self.features["delta"]
    
    def _compute_delta_delta(self, audio_track, audio_rate):
        """Delta are a set of coefficients defined as the local derivatives of the delta (or logenergy) 
        coefficients, and so the second derivative of the MFCCs.
        The function returns a vector of size (12, NFRAMES) = (12,431)."""
        
        if "delta-delta" not in self.features: 
            self.features["delta-delta"] = librosa.feature.delta(self._compute_mfcc(audio_track, audio_rate), order=2)
        return self.features["delta-delta"]
    
    def _compute_energy(self, audio_track):
        """If s(n) is the signal in the time domain, with n being the discrete sampling time, the root mean 
        square energy of a generic frame i is:       E(s_i) = SQRT((1/N)*sum_n s(n)**2)  
        where the sum goes over all the samples in the frame.
        The function returns a vector of size NFRAMES = 431, with one value for each frame."""
        
        if "energy" not in self.features: 
            self.features["energy"] = librosa.feature.rms(audio_track)
        return self.features["energy"]
    
    def _compute_delta_energy(self, audio_track):
        """First derivative of the energy vectors.
        The function returns a vector of size NFRAMES = 431, with one value for each frame."""
        
        if "delta-energy" not in self.features: 
            self.features["delta-energy"] = librosa.feature.delta(self._compute_energy(audio_track), order=1)
        return self.features["delta-energy"]
    
    def _compute_delta_delta_energy(self, audio_track):
        """Second derivative of the energy vectors. 
        The function returns a vector of size NFRAMES = 431, with one value for each frame."""
        
        if "delta-delta-energy" not in self.features: 
            self.features["delta-delta-energy"] = librosa.feature.delta(self._compute_energy(audio_track), order=2)
        return self.features["delta-delta-energy"]
        
    def Play(self):
        """ Exploit the functionality of IPython to run the audio of the clip """
        with self.audio as audio:
            play = IPython.display.Audio(audio.data, rate=audio.rate)
        return play
            
    def DisplayWave(self, ax=None):
        """ Plot the raw signal stored in the clip """
        with self.audio as audio:
            if ax is None: fig, ax = plt.subplots(1,1, figsize=(8,3))
            librosa.display.waveplot(audio.raw, sr=audio.rate, ax=ax, alpha=0.4)
            ax.set_title(r"$\bf{Audio Signal:}$  " + self.category + ' - ' + self.clipname)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
        return ax
    
    def DisplaySpectrogram(self, ax=None, cbar=True):
        """ Plot the spectrogram of raw signal stored in the clip """
        with self.audio as audio:
            # We keep default parameters(n_fft=2048, hop_length=512, win_length=2048)
            X = librosa.stft(audio.raw)
            Xdb = librosa.amplitude_to_db(abs(X))
            if ax is None: _, ax = plt.subplots(1,1, figsize=(9,3))
            ss = librosa.display.specshow(Xdb, sr=audio.rate, ax=ax, x_axis='time', y_axis='log', cmap='RdBu_r')
            if cbar: plt.colorbar(ss)
            ax.set_title(r"$\bf{Spectogram:}$  " + self.category + ' - ' + self.clipname)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (dB)")
        return ax
    
    def DisplayMelSpectrogram(self, ax=None, cbar=True):
        """ Plot the spectrogram in the Mel Scale of raw signal stored in the clip """
        with self.audio as audio:
            # We keep default parameters(n_fft=2048, hop_length=512, win_length=2048)
            X = librosa.feature.melspectrogram(audio.raw, sr=audio.rate)
            Xdb = librosa.amplitude_to_db(abs(X))
            if ax is None: _, ax = plt.subplots(1,1, figsize=(9,3))
            ss = librosa.display.specshow(Xdb, sr=audio.rate, ax=ax, x_axis='time', y_axis='log', cmap='RdBu_r')
            if cbar: plt.colorbar(ss)
            ax.set_title(r"$\bf{Mel Spectogram:}$  " + self.category + ' - ' + self.clipname)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (dB)")
        return ax


def Collect_Clips(data_dir, label_map, macro_map, leave_tqdm=True):
    df = pd.DataFrame(None, columns=["macro-category", "label", "clip"])
    for i, file in enumerate(tqdm(sorted(os.listdir(data_dir)), desc='Collecting clips in the folder', unit='clips', leave=leave_tqdm)):
        num_label = int(file.split('-')[3].split('.')[0])
        cat_label = label_map[num_label]
        macro_label = macro_map(num_label)
        track = Clip(data_dir + file, label_map)
        df.loc[i] = macro_label, cat_label, track
    return df


def initialize_folder(folder_name):
    
    # If the folder already exists, delete it
    if os.path.exists(folder_name): shutil.rmtree(folder_name)
    
    os.mkdir(folder_name)
    


