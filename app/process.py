from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import librosa
import numpy as np
import csv
import os
from app import APP_ROOT
from app.normalization import normalize


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
    # model_path = os.path.join(APP_ROOT, 'static/Logistic_regression.pickle')
    # with open(model_path, 'rb') as f:
    #     model = pickle.load(f)

    model_path = os.path.join(APP_ROOT, 'static/xgboost.pickle')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Extract features from sound
    featureList = extractFeature(path)

    # Normalize 'featureList'
    featureList = normalize(featureList)
    predictions = model.predict([featureList])
    # print(features)
    
    return features + predictions.tolist()
