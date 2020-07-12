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
from keras.layers import Dot, GaussianNoise, SpatialDropout1D,Subtract, Flatten, Bidirectional, GRU, Masking
import keras.backend as K
import tensorflow as tf

from utils import *


print('GPU avialible: {}'.format(tf.config.list_physical_devices('GPU')))
pd.options.display.width = 500

import pandas as pd

n_embedding_dims = 50
drop_rate = .3
num_filters = 50
kernal_size = 3
padding = 'pre'
truncating = 'pre'
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
#

print('loading embeddings ... ')
emb_dict = joblib.load('models/emb_dict.jbl')

def data_gen(docs, n_batches=10, batch_size=10):
    n_docs = len(docs)
    sample_index = np.arange(batch_size, n_docs- (batch_size + int(batch_size/2)))
    for _ in range(n_batches):
        index = np.random.choice(sample_index, batch_size)
        in_box_docs = np.array(docs)[index]
        out_box_docs =  nn_doc_sampler(in_box_docs, docs)
        in_box_seqs = tokenizer.texts_to_sequences(in_box_docs)
        ## data augmentation using utils to change sequence inputs
        in_box_seqs_aug = augment_sequences(tokenizer.texts_to_sequences(augment_text(in_box_docs)))
        in_box_array = np.array(pad_sequences(in_box_seqs,**params))
        in_box_array_aug = np.array(pad_sequences(in_box_seqs_aug,**params))
        out_box_array = np.array(pad_sequences(tokenizer.texts_to_sequences(out_box_docs), **params))
        x = np.concatenate([in_box_array_aug, in_box_array], axis=0)
        x_aux = np.concatenate([in_box_array, out_box_array], axis=0)
        y = np.array([1] * in_box_array.shape[0] + [0] * out_box_array.shape[0])
        assert x_aux.shape[0] == x.shape[0]
        assert y.shape[0] == x.shape[0]
        yield [x, x_aux], y
(x, x_aux), y = next(data_gen(docs))
print(x.shape, x_aux.shape, y.shape)

embedding_array = np.zeros((vocab_size +1,n_embedding_dims ))
missing = 0
found = 0
for (word, i) in tokenizer.word_index.items():
    try:
        embedding_array[i, :] = emb_dict[word]
        missing += 1
    except:
        found += 1
print('missing {}'.format(missing/(missing +found)))

def get_model():

    inputs = Input(shape=(max_len,),  name='input')
    embedding_layer =  Embedding(vocab_size+ 1, n_embedding_dims,
                                 weights=[embedding_array],
                                 input_length=max_len,
                                 trainable=False,
                                 mask_zero=True)(inputs)

    inputs_aux = Input(shape=(max_len,),  name='input_aux')
    embedding_layer_aux =  Embedding(vocab_size+ 1, n_embedding_dims,
                                 weights=[embedding_array],
                                 input_length=max_len,
                                 trainable=False,
                                 mask_zero=True)(inputs_aux )

    sub = Subtract()([embedding_layer, embedding_layer_aux])
    recur = Bidirectional(GRU(max_len, return_sequences=False, recurrent_dropout=drop_rate))(sub)
    drop = Dropout(drop_rate)(recur)
    act1 = Activation('relu', name = "activ_solo")(drop)  ## weird that the activation is stack on a dropout layr
    outputs = Dense(1, activation='sigmoid', name='sigmoidOutput')(act1)
    model = Model(inputs=[inputs,inputs_aux],  outputs=outputs, name='MatchingNamesClassifier')
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = get_model()
print(model.summary())

weights_path = "models/weights_best.hdf5"

try:
    model.load_weights(weights_path )
    print('best weights loaded from {}'.format(weights_path ))
except:
    print('building new model from scatch ...')


#save only the best weights
checkpoint = ModelCheckpoint(weights_path ,mode='max' ,monitor='val_accuracy', verbose=1, save_best_only=True)

learning_rate = .0004

#update the optimizer learning rate
K.set_value(model.optimizer.lr,learning_rate)

lrPlateauReductionFactor = .5
lrMin = 0.000001

# reduces learning rate on performance platu
lrCheckPoint = ReduceLROnPlateau(monitor =  'val_loss', factor=lrPlateauReductionFactor, min=lrMin)

# stops training whenmodel fails to improve
esm =  EarlyStopping(patience=5, monitor='val_accuracy',mode='max')


n_epochs = 500
n_steps_per_epoch = 100
batch_size = 32

train_docs, test_docs = train_test_split(docs)
train_docs.sort()
test_docs.sort()

train_gen = data_gen(train_docs, n_batches=n_epochs * n_steps_per_epoch, batch_size=batch_size)
test_gen = data_gen(test_docs, n_batches=n_epochs * n_steps_per_epoch, batch_size=batch_size * 2)
# Place tensors on the CPU
with tf.device('/GPU:1'):
    model.fit(train_gen,
                    epochs=n_epochs,
                    steps_per_epoch= n_steps_per_epoch,
                    validation_data = test_gen,
                    validation_steps = n_steps_per_epoch,
                    callbacks=[esm, lrCheckPoint, checkpoint], shuffle=True)

    test_names = ['Brautigan, Richard', 'Richard Branson',"Louis Farrakhan", "Louis Farrakhan" ]
    test_matches = ['Brauttigan Richard', 'Richard Branson',"Louis Farrakhan Sr.", "Louis Fara"]
    x_test = np.array(pad_sequences(tokenizer.texts_to_sequences(preprocess_texts(test_names)), **params))

    x_test_aux = np.array(pad_sequences(tokenizer.texts_to_sequences(preprocess_texts(test_matches)),**params))
    print(x_test )
    print(x_test_aux)
    print('testing model ...')
    print('test data shapes:', x_test.shape, x_test_aux.shape)
    p = model.predict([x_test , x_test_aux ]).flatten()
    print(p)


best_acc = max(model.history.history['val_accuracy'])
print('best acc {} '.format(best_acc))
if all((p[0] > p[1], best_acc > .9, p[2] >p[3])):
    # model persitance
    model_path = 'models/model.h5'
    model.save(model_path)
    print('saving to {}'.format(model_path))

    tokenizing_path = 'models/tokenizer.jbl'
    joblib.dump(tokenizer,tokenizing_path )
    print('saving tokenizer to {}'.format(tokenizing_path))
else:
    print('skipping save due to lack of performance')
print('end')
