import mne

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from IPython.display import clear_output
import os

from scipy import io
from scipy.stats import entropy
import pywt

def build_dataframe(data_path):
    ext_annot = pd.read_excel(data_path + 'External_Annotations.xlsx')
    videos_annot = ext_annot.drop(['UserID', 'VideoID'], axis=1)\
                    .merge(pd.read_excel(data_path + 'Video_List.xlsx')[['Video_Number', 'Category']], 
                           on='Video_Number')

    users_big5 = pd.read_excel(data_path + 'Participants_Personality.xlsx', 
                               sheet_name='Personalities').T.reset_index()
    users_big5.columns = users_big5.iloc[0]
    users_big5 = users_big5[1:]

    df = pd.DataFrame({'UserID': [], 'Video_Number': [], 'EEG': [], 'Video_Rating': []})

    for f in sorted([data_path + x for x in os.listdir(data_path) if 'Data_Preprocessed_P' in x]):
        UserID = f.split('_')[-1].split('.')[0].replace('P', '')

        data = io.loadmat(f)
        eegs = data['joined_data'][0].tolist()
        video_ratings = data['labels_selfassessment'][0].tolist()

        df0 = pd.DataFrame({'UserID': len(video_ratings)* [int(UserID)], 
                            'Video_Number': list(range(1,len(video_ratings) + 1)), 
                            'EEG': eegs, 'Video_Rating': video_ratings})

        df = pd.concat([df, df0], axis=0)

    df = df.reset_index(drop=True)
    df.UserID = df.UserID.astype(int)
    df.Video_Number = df.Video_Number.astype(int)
    df = df.merge(videos_annot[['Video_Number', 'Category']].drop_duplicates().rename({'Category':'Video_category'}, axis=1), 
                  on='Video_Number')

    kkkl = ['arousal', 'valence', 'dominance', 'liking', 'familiarity', 'neutral', 'disgust', 'happiness', 'surprise', 'anger', 'fear', 'sadness']
    for i in range(len(kkkl)):
        df['Video_' + kkkl[i]] = [x[0][i] if x[0].shape[0] == 12 else None for x in df.Video_Rating]

    df.drop('Video_Rating', axis=1, inplace=True)

    del kkkl

    df = df.merge(users_big5, on='UserID')
    df = df[df.Video_Number < 17].reset_index(drop=True)

    df.EEG = df.EEG.apply(lambda x: x[384:, :14])
    df = df[[not bool(x[np.isnan(x)].shape[0]) for x in df.EEG]]
    df.reset_index(drop=True, inplace=True)

    return df

def scale_eeg(eegmat, low_filter=1, high_filter=60):
    scaler = MinMaxScaler()
    eegmat = scaler.fit_transform(eegmat.T) * 10e-5
    
    sampling_freq = 128  # in Hertz
    info = mne.create_info(['AF3', 'F7', 'F3', 'FC5', 'T7', 
                            'P7', 'O1', 'O2', 'P8', 'T8', 
                            'FC6', 'F4', 'F8', 'AF4'], sfreq=sampling_freq, 
                          ch_types= 14*['eeg'])

    raw = mne.io.RawArray(eegmat, info)
    raw.set_montage('standard_1020')

    raw.filter(low_filter, high_filter) 
    clear_output(wait=False)
    
    return raw.get_data()

def Energy(signal):
    return np.diag(np.dot(signal, signal.T))

def discrete_wavelet_transform(signal, feature, windowsize=4, sampling_rate=128):
    samples = int(sampling_rate * windowsize / 2)
    n = signal.shape[1]
    
    scaler = MinMaxScaler()
    
    if feature == "entropy":
        for i in range((samples), (n - samples), samples):
            lowpass_signal, noise=pywt.dwt(signal[:, i-samples:i + samples], 'db4')
            
            for j in range(5):
                lowpass_signal, highpass_signal = pywt.dwt(lowpass_signal, 'db4')
                highpass_signal=scaler.fit_transform(highpass_signal.T).T
                
                if j==0:
                    if i == samples:
                        gamma=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        gamma=np.concatenate((gamma, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)
                if j==1:
                    if i == samples:
                        beta=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        beta=np.concatenate((beta, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)                
                if j==2:
                    if i == samples:
                        alpha=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        alpha=np.concatenate((alpha, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)
                if j==3:
                    if i == samples:
                        theta=entropy(highpass_signal.T).reshape(1,-1).T
                    else:
                        theta=np.concatenate((theta, 
                                    entropy(highpass_signal.T).reshape(1,-1).T), axis=1)
                if j==4:
                    if i == samples:
                        delta=entropy(highpass_signal).reshape(1,-1).T
                    else:
                        delta=np.concatenate((delta, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
    
    elif feature == "energy":
        for i in range((samples), (n-samples), samples):
            lowpass_signal, noise=pywt.dwt(signal[:,i-samples:i+samples], 'db4')
            
            for j in range(5):
                lowpass_signal, highpass_signal = pywt.dwt(lowpass_signal, 'db4')
                
                if j==0:
                    if i == samples:
                        gamma=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        gamma=np.concatenate((gamma, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
                if j==1:
                    if i == samples:
                        beta=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        beta=np.concatenate((beta, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)                
                if j==2:
                    if i == samples:
                        alpha=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        alpha=np.concatenate((alpha, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
                if j==3:
                    if i == samples:
                        theta=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        theta=np.concatenate((theta, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1) 
                if j==4:
                    if i == samples:
                        delta=Energy(highpass_signal).reshape(1,-1).T
                    else:
                        delta=np.concatenate((delta, 
                                    Energy(highpass_signal).reshape(1,-1).T), axis=1)
            
    gamma=scaler.fit_transform(gamma.T).T
    beta=scaler.fit_transform(beta.T).T
    alpha=scaler.fit_transform(alpha.T).T
    theta=scaler.fit_transform(theta.T).T
    delta=scaler.fit_transform(delta.T).T
    
    return np.array([gamma, beta, alpha, theta, delta])

def zero_padding(EEG_list, total_length):
    data_list = []
    
    for i in range(len(EEG_list)):
        EEG_pad = np.pad(EEG_list[i], ((0,0),(0,0), (0, total_length-EEG_list[i].shape[2])), 'constant')
        data_list.append(EEG_pad)
        
    return data_list

