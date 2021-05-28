import os
import pickle
from app import APP_ROOT


def normalize(featureList):
    list_path = os.path.join(APP_ROOT, 'static/norm_list.pickle')

    # Load norm_dict
    with open(list_path, 'rb') as f:
        norm_list = pickle.load(f)

    for i in range(featureList):
        featureList[i] = (featureList[i] - norm_list[i][0]) / norm_list[i][1]

    return featureList