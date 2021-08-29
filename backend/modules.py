from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features

import speech_recognition as sr
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

recognizer = sr.Recognizer()

def audio_features(audio_file_path : str):
    waveform = Waveform(path=audio_file_path)
    feature_df = extract_features(waveforms=[waveform], components_list=components_list, statistics_list=statistics_list)
    return feature_df

def audio_to_text(audio_file_path : str):

    try:
        sound = AudioSegment.from_mp3(audio_file_path)
        sound.export("temp.wav", format="wav")
        with sr.AudioFile('temp.wav') as source:
            audio = recognizer.record(source)                  
            text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return " "