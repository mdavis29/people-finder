 ## Deep Learning Sentiment Analysis
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import joblib
import os
import requests
from pathlib import Path
import json
import cloudpickle
import pickle
from sklearn.model_selection import train_test_split
import re
from multiprocessing import cpu_count, Pool
import nltk
from nltk.corpus import stopwords
from itertools import chain

import plotly.express as px
import plotly.graph_objects as go


import tensorflow as tf
import re
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import skipgrams
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Dropout, Reshape, Activation, Input, Flatten, Dense, RepeatVector, BatchNormalization, LSTM, RepeatVector, MaxPooling1D,GlobalMaxPool1D, Conv1D, GRU, Bidirectional, Concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dot, GaussianNoise, SpatialDropout1D,Subtract, Flatten, Bidirectional, GRU
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.sequence import make_sampling_table, skipgrams
from utils import *


print('GPU avialible: {}'.format(tf.config.list_physical_devices('GPU')))
pd.options.display.width = 500

import pandas as pd

n_embedding_dims = 50
drop_rate = .3
num_filters = 50
kernal_size = 2
padding = 'post'
truncating = 'post'
noise_level = .1

## load data
docs = []
for p in os.listdir('data'):
    with open('data/'+p, 'r') as f:
        docs = docs + f.readlines()
docs = [d.strip() for d in docs]
docs = [d for d in docs if len(d) > 10]
docs.sort()
docs =  preprocess_texts(docs)
print(docs[0:100])
print('loaded {} docs'.format(len(docs)))
# fit the tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(docs)
vocab_size = len(list(tokenizer.index_word.items()))
max_len = max([len(d) for d in docs])
params={'maxlen':max_len, 'padding':padding, 'truncating':truncating}
print('max_len {0} vocab_size {1}'.format(max_len, vocab_size))


window_size = 2


def get_model():

    inputs = Input(shape=(1,),  name='input')
    embedding_layer =  Embedding(vocab_size+ 1, n_embedding_dims,
                                 input_length=1,
                                 trainable=True,
                                 name='emb')(inputs)

    inputs_aux = Input(shape=(1,),  name='input_aux')
    embedding_layer_aux =  Embedding(vocab_size+ 1, n_embedding_dims,
                                 input_length=1,
                                 trainable=True,
                                 name='emb_aux')(inputs_aux )

    merge = Dot(axes=1)([embedding_layer, embedding_layer_aux])
    flat = Flatten()(merge)
    outputs = Dense(1, activation='sigmoid', name='sigmoidOutput')(flat )
    model = Model(inputs=[inputs,inputs_aux],  outputs=outputs, name='MatchingNamesClassifier')
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = get_model()
print(model.summary())

weights_path='models/skgrams.hdf5'
#save only the best weights
checkpoint = ModelCheckpoint(weights_path ,mode='max' ,monitor='accuracy', verbose=1, save_best_only=True)

learning_rate = .0004

#update the optimizer learning rate
K.set_value(model.optimizer.lr,learning_rate)

lrPlateauReductionFactor = .5
lrMin = 0.000001

# reduces learning rate on performance platu
lrCheckPoint = ReduceLROnPlateau(monitor =  'loss', factor=lrPlateauReductionFactor, min=lrMin)

# stops training whenmodel fails to improve
esm =  EarlyStopping(patience=2, monitor='accuracy',mode='max')

n_epochs = 500

batch_size = 32
n_steps_per_epoch = len(docs)


def data_gen(docs, n_batches = 10):
    seqs = tokenizer.texts_to_sequences(docs)
    for _ in range(n_batches):
        for seq in seqs:
            grams, y=  skipgrams(seq, window_size=window_size,  vocabulary_size=vocab_size)
            x = np.array([v[0] for v in grams])
            x_aux = np.array([v[1] for v in grams])
            yield [x, x_aux], np.array(y)
(x, x_aux), y = next( data_gen(docs, n_batches = 10))
print(x.shape, x_aux.shape, y.shape)

gen = data_gen(docs, n_batches=n_epochs* n_steps_per_epoch )


# Place tensors on the CPU
with tf.device('/GPU:1'):
    pass
    model.fit(gen,
                epochs=n_epochs,
                steps_per_epoch= n_steps_per_epoch,
                validation_steps = n_steps_per_epoch,
                callbacks=[esm, lrCheckPoint, checkpoint], shuffle=True)
d = {}
embeddings = model.get_layer(name='emb_aux').get_weights()[0]
for (word, i) in tokenizer.word_index.items():
    d[word] = embeddings[i, :]
joblib.dump(d, 'models/emb_dict.jbl')
print('completed')
