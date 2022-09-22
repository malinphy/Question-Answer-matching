
!pip install --quiet scann
# !pip install --quiet datasets
# !pip install --quiet pipreqsnb
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import *
import tensorflow_hub as hub 

import scann
# from data_loader import data_loader
# from negative_maker import negative_maker
from model import model

use_hub = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") ## universal sentence encoder model

from google.colab import drive
drive.mount('/content/drive')
triplet_model = model()
triplet_model.load_weights('drive/MyDrive/Colab Notebooks/quick_response/triplet_model_weights.h5')

test_df = pd.read_csv('test_df.csv')

use_emb =  triplet_model.get_layer('sentence_encoder')
saving_path = 'drive/MyDrive/Colab Notebooks/quick_response/scann_save'
searcher = scann.scann_ops_pybind.load_searcher(saving_path)

def prediction(test_sentence):
    test_quest_emb = np.array(use_emb(([test_sentence]))).reshape(1,512)

    index, distance = searcher.search(test_quest_emb.ravel())

    return test_df['first_answer'][index[0]]


prediction('What\'s the difference between a bush, a shrub, and a tree?')

prediction('Is there any difference between a bush, a shrub, and a tree?')