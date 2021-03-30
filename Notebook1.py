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
import soundfile as sf
import shutil
import re

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




def get_augmented_data(augmented_folder):
    
    augmented_data = pd.DataFrame(None, columns=["macro-category", "label", "clip"])
    for data_folder in os.listdir(augmented_folder):

        clip_fold = augmented_folder+data_folder+'/'
        augmented_data = pd.concat([augmented_data, Collect_Clips(clip_fold, get_label_map(), macro_categories_map, leave_tqdm=False)])

    augmented_data.reset_index(drop=True, inplace=True)

    return augmented_data




def get_zero_windows(vector, window_size):
    """This function takes in input a numerical 1D array and search in it for fixed sized sequences of zeros. 
    Then, it will return a vector a vector of booleans of the same size of the input one, to be used to
    select such windows."""
    
    temp = vector.copy()
    # For conveniency, convert all the non-zero elements of the array into ones
    temp[temp>(1e-6)] = 1
    temp_str = ''.join(str(int(x)) for x in temp)
    del temp
    it = re.finditer(pattern='0{'+str(window_size)+'}', string=temp_str)
    
    window_pattern = [False]*len(vector)
    
    for index in [x.start() for x in it]:
        window_pattern[index:index+window_size] = [True]*window_size
        
    return window_pattern





def Construct_Vector_Features(clip, stat=['mean'], silence_elimination=0):
    """Given a Clip object, this function return an array with all the summarized features per frame. 
    By default, the distributions are represented with their mean, but other statistical measures, 
    can be selected. If you want to drop windows of zeros in those vectors, you insert the time in
    seconds in the parameter silence_elimination."""
    
    if silence_elimination != 0:
        # Select a window based on the energy values
        silent_windows = get_zero_windows(vector=clip.features['energy'].T, 
                                          window_size=int(431//5*silence_elimination))
    
    vector = []
    feat_dict = clip.features.copy()

    # Single features
    for feat in ['spectral centroid', 'spectral rolloff', 'spectral bandwidth', 'zero-crossing rate', 
                 'energy', 'delta-energy', 'delta-delta-energy']:

        if silence_elimination != 0: 
            if feat_dict[feat].shape[0]!=431: feat_dict[feat].flatten()[silent_windows] = np.nan
                    
        if 'mean' in stat: 
            vector += [np.nanmean(feat_dict[feat])]
        if 'std'  in stat: 
            vector += [np.nanstd(feat_dict[feat])]
        if 'median' in stat: 
            vector += [np.nanmedian(feat_dict[feat])]
        
    # N-dimensional features
    for feat in ['mfcc', 'chromagram', 'delta', 'delta-delta']:
        
        if silence_elimination != 0: feat_dict[feat][:,silent_windows] = np.nan
        
        if 'mean' in stat: 
            vector += list(np.nanmean(feat_dict[feat], axis=1))
        if 'std'  in stat: 
            vector += list(np.nanstd(feat_dict[feat], axis=1 ))
        if 'median' in stat: 
            vector += list(np.nanmedian(feat_dict[feat], axis=1))
        
    return np.array(vector)






def compute_macro_conf_mat(confusion_matrix):
    """Given the confusion matrix for the 50 classes, this function return the corresponding mapped confusion 
    matrix of the macro-categories."""
    
    macro_confusion_matrix = np.zeros((5,5))
    
    # Map micro into macro categories
    animals  = [5, 13, 16, 18, 25, 29, 30, 34, 37, 39]
    natural  = [7, 14, 15, 35, 36, 38, 43, 44, 48, 49]
    humans   = [1, 2,   9, 12, 17, 21, 24, 32, 41, 42]
    domestic = [3, 10, 11, 19, 20, 26, 31, 33, 46, 47]
    urban    = [0, 4,   6,  8, 22, 23, 27, 28, 40, 45]
    
    macro_categories = [animals, natural, humans, domestic, urban]
    
    for i in range(5):
        for j in range(5):
            for cat in macro_categories[j]:
                macro_confusion_matrix[i,j] += sum(confusion_matrix[macro_categories[i], cat])

    return macro_confusion_matrix


def extract_features(dataset, stat=['mean'], silence_window=0, csv_filename=None):
    """Given a dataset containing all the instances of the class Clip, this function construct a pandas 
    dataframe in which each row is composed by the vector of features of one clip, and its corresponding
    label. Then, it can save the result in a csv file."""
    
    n_features = 55
    
    features_df = pd.DataFrame(columns=[i for i in range(n_features*len(stat))])
    
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        
        # Compute all the features useful for the classification
        sample = row['clip']
        sample.Compute_Features('all')
        
        features_df.loc[index] = Construct_Vector_Features(sample, stat, silence_window)
    
    features_df['label'] = dataset['clip'].apply(lambda x: x.category)
    features_df['clipname'] = dataset['clip'].apply(lambda x: x.clipname)
    
    if csv_filename is not None: features_df.to_csv(csv_filename)
        
    return features_df



def Add_Normal_Noise(data):
    """Generate some random gaussian noise regularized by a proper factor and add it to the signal."""
    noise = np.random.randn(len(data))
    noise_factor = 0.05# np.random.uniform(low=0.0, high=0.05)
    augmented_data = data + noise_factor*noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def Shift_Time(data):
    """Apply a random time shift to the signal, padding with zeros the shifted part."""
    # Max shift = 1/3 of the clip
    shift = np.random.randint(len(data)//3)
    # Select at random if shifting to the right or to the left
    direction = np.random.randint(0, 2)
    if direction == 1: shift = -shift        
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def Pitch_Shift(data, rate):
    """Raise or low the pitch of the audio sample."""
    pitch_factor = 2#np.random.uniform(low=-3.0, high=3.0)
    return librosa.effects.pitch_shift(data, rate, pitch_factor)

def Pitch_Shift2(data, rate):
    """Raise or low the pitch of the audio sample."""
    pitch_factor = -2#np.random.uniform(low=-3.0, high=3.0)
    return librosa.effects.pitch_shift(data, rate, pitch_factor)

def Time_Stretch(data):
    """Slow down or speed up the audio sample (keeping the pitch unchanged)."""
    speed_factor = 0.70#np.random.uniform(low=0.5, high=1.5)
    return librosa.effects.time_stretch(data, speed_factor)

def Time_Stretch2(data):
    """Slow down or speed up the audio sample (keeping the pitch unchanged)."""
    speed_factor = 1.20#np.random.uniform(low=0.5, high=1.5)
    return librosa.effects.time_stretch(data, speed_factor)
    



def Augment_Clips(data, augmented_folder):
    """For every clip in the original dataset, we will construct 4 additional new audio files, corresponding
    to the 4 kind of augmentation that we implemented: noise addition, time shifting, pitch shifting and time
    stretching."""
    
    initialize_folder(augmented_folder) 
    
    augmented_folder_1 = augmented_folder + 'original/'
    augmented_folder_2 = augmented_folder + 'noisy/'
    augmented_folder_3 = augmented_folder + 'shifted/'
    augmented_folder_4 = augmented_folder + 'pitch_shifted/'
    augmented_folder_5 = augmented_folder + 'stretched/'
    augmented_folder_6 = augmented_folder + 'pitch_shifted2/'
    augmented_folder_7 = augmented_folder + 'stretched2/'
    initialize_folder(augmented_folder_1) 
    initialize_folder(augmented_folder_2) 
    initialize_folder(augmented_folder_3) 
    initialize_folder(augmented_folder_4) 
    initialize_folder(augmented_folder_5)
    initialize_folder(augmented_folder_6)
    initialize_folder(augmented_folder_7) 
    
    for index, row in tqdm(data.iterrows(), total=len(data), leave=False,
                           desc='Collecting clips in folder ' + augmented_folder, unit='clips'):
        
        with row['clip'].audio as audio:
            
            filename = row['clip'].clipname
            
            sf.write(file=augmented_folder_1+filename, data=audio.raw, samplerate=audio.rate, subtype='PCM_24')
        
            sf.write(file=augmented_folder_2+filename, data=Add_Normal_Noise(audio.raw), 
                     samplerate=audio.rate, subtype='PCM_24')
            
            sf.write(file=augmented_folder_3+filename, data=Shift_Time(audio.raw), 
                     samplerate=audio.rate, subtype='PCM_24')
            
            sf.write(file=augmented_folder_4+filename, data=Pitch_Shift(audio.raw, audio.rate), 
                     samplerate=audio.rate, subtype='PCM_24')
    
            sf.write(file=augmented_folder_5+filename, data=Time_Stretch(audio.raw), 
                     samplerate=audio.rate, subtype='PCM_24')

            sf.write(file=augmented_folder_6+filename, data=Pitch_Shift2(audio.raw, audio.rate), 
                     samplerate=audio.rate, subtype='PCM_24')

            sf.write(file=augmented_folder_7+filename, data=Time_Stretch2(audio.raw), 
                     samplerate=audio.rate, subtype='PCM_24')
            
            
     
