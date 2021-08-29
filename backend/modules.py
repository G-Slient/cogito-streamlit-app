import pandas as pd
import speech_recognition as sr
import joblib
import os
import numpy as np
import librosa
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
from pydub import AudioSegment

import warnings
warnings.simplefilter('ignore')

components_list = [
    'mfcc', 
    'spectral_flux',
    'spectral_slope',
    'spectral_centroid',
    'spectral_spread',
    'spectral_skewness',
    'spectral_kurtosis',
    'spectral_rolloff',
    'shannon_entropy_slidingwindow',
    'rms',
    'bark_spectrogram',
    'morlet_cwt',
    'chroma_stft', 
    'chroma_cqt',
    'chroma_cens',
    'spectral_entropy',
    'spectral_flatness',
    'loudness_slidingwindow',
    'zerocrossing_slidingwindow',
    'intensity',
    'crest_factor',
    'f0_contour',
    'lpc',
    'lsf',
    'formants_slidingwindow',
    'kurtosis_slidingwindow',
    'log_energy_slidingwindow',
    'ppe',
    'loudness',
    'shannon_entropy',
    'hnr',
    'dfa',
    'log_energy',
    'zerocrossing',
    'f0_statistics',
    'jitters',
    'shimmers',
    'formants',
]

statistics_list = ['mean', 'std', 'first_derivative_mean', 'first_derivative_std']

labels = ['anger' ,'disgust', 'fear' ,'joy', 'neutral' ,'sadness', 'surprise']



def audio_features(audio_file_path : str):
    waveform = Waveform(path=audio_file_path)
    feature_df = extract_features(waveforms=[waveform], components_list=components_list, statistics_list=statistics_list)
    return feature_df

def audio_to_text(audio_file_path : str):

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)                  
            text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return " "

def load_features()-> list:

    feat = joblib.load(os.path.join("model","features_list.pkl"))
    return feat

def calFileSize(audio_file_path:str):
     return os.stat(audio_file_path).st_size

def calAudioDuration(audio_file_path:str):
    return librosa.get_duration(filename=audio_file_path)

def load_model():
    model = joblib.load(os.path.join("model","model.pkl"))
    return model

def getPredictions(model,df,feat):

    output_dic = {}
    try:
        #process the null, infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        null_cols = list(df.columns[df.isnull().any()])
        df[null_cols] = df[null_cols].astype(float)

        preds = model.predict(df[feat])
        res = dict(zip(labels, preds[0]))
        output_dic['probabilities'] = res

        emotion = max(res, key= lambda x: res[x])
        output_dic['Emotion'] = emotion.title()
        output_dic['Confidence'] = round(output_dic['probabilities'][emotion],2)*100
        output_dic['status'] = 200

        return output_dic
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print("Error in getPredictions")
        output_dic['status'] = 202
        return output_dic
