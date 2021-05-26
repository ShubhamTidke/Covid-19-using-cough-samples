from sklearn.linear_model import LogisticRegression
import pickle
import librosa
import numpy as np
import csv
import os
from app import APP_ROOT


def extractFeature(path):
    path = os.path.join(APP_ROOT, 'audio.wav')
    y, sr = librosa.load(path, mono=True)

    # Extracting features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Appends the features to 'featureList'
    featureList = []
    for i in [chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc]:
        featureList.append(np.mean(i))

    return featureList


def prognosis(path, features):
    model_path = os.path.join(APP_ROOT, 'static/Logistic_regression.pickle')
    with open(model_path, 'rb') as f:
        lr = pickle.load(f)
    
    featureList = extractFeature(path)
    # print(featureList)
    predictions = lr.predict([featureList])

    # print(type(predictions.tolist()))
    # print(type(features))
    
    return features + predictions.tolist()
