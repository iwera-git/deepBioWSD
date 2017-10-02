import random
import sys
import keras.backend as K
import numpy as np
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
import operator
from keras.layers import Input, Embedding, AveragePooling1D, MaxPooling1D, Flatten, Dense, Dropout, Merge, Highway, Activation, Reshape
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import regularizers


def create_embedding(concept_dic, embeddings_file, EMBEDDING_DIM, MAX_SENSE_LENGTH = 5, PRE_TRAINED=True, UPDATABLE=True):

    if PRE_TRAINED:
        file_reader = open(embeddings_file, "r")
        concept_embeddings = {}
        for line in file_reader:
            values = line.split()
            concept = values[0]
            concept_embeddings[concept] = np.asarray(values[1:], dtype='float32')
        file_reader.close()

        print('Loaded', len(concept_embeddings), 'concept vectors.')

        embedding_matrix = np.zeros((len(concept_dic) + 1, EMBEDDING_DIM)) - 300.0
        for concept, index in concept_dic.items():
            embedding_vector = concept_embeddings.get(concept)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        embedding_layer = Embedding(input_dim=len(concept_dic) + 1,
                                    output_dim=EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SENSE_LENGTH,
                                    trainable=UPDATABLE)
    else:
        embedding_layer = Embedding(input_dim=len(concept_dic) + 1,
                                    output_dim=EMBEDDING_DIM,
                                    input_length=MAX_SENSE_LENGTH)

    return embedding_layer


def build_network(concept_dic, embeddings_file, EMBEDDING_DIM=100, MAX_SENSE_LENGTH = 5, CONTEXT_WINDOW_SIZE = 5,
                  PRE_TRAINED=True, UPDATABLE=True,
                  dropout_rate=0.3,
                  hidden_activation="relu", highway_activation="sigmoid", output_activation="linear",
                  optimizer="adam", print_model=False):

    INPUTS = []
    LEFT_RIGHT_CENTER = []

    embedding_layer = create_embedding(concept_dic, embeddings_file,
                                       EMBEDDING_DIM, MAX_SENSE_LENGTH, PRE_TRAINED, UPDATABLE)

    for i in range(2 * CONTEXT_WINDOW_SIZE + 1):
        """Creating network's pipes one-by-one (from left to right)"""

        context_term_input = Input(shape=(MAX_SENSE_LENGTH,), dtype='int32')
        INPUTS.append(context_term_input)

        context_term_embedding = embedding_layer(context_term_input)

        pipe = MaxPooling1D(pool_size=MAX_SENSE_LENGTH)(context_term_embedding)
        pipe = Flatten()(pipe)
        LEFT_RIGHT_CENTER.append(pipe)

    left = Merge(mode='max')(LEFT_RIGHT_CENTER[0:CONTEXT_WINDOW_SIZE])
    left_dense = Dense(units=EMBEDDING_DIM, activation=hidden_activation)(left)
    left_dense_dropout = Dropout(dropout_rate)(left_dense)

    right = Merge(mode='max')(LEFT_RIGHT_CENTER[CONTEXT_WINDOW_SIZE:CONTEXT_WINDOW_SIZE * 2])
    right_dense = Dense(units=EMBEDDING_DIM, activation=hidden_activation)(right)
    right_dense_dropout = Dropout(dropout_rate)(right_dense)
    
    context = Merge(mode='max')([left_dense_dropout, right_dense_dropout])

    centre = LEFT_RIGHT_CENTER[-1]
    #centre_dense = Dense(units=EMBEDDING_DIM, activation=hidden_activation)(centre)
    #centre__dense_dropout = Dense(units=EMBEDDING_DIM, activation=hidden_activation)(centre_dense)
    
    merge_instance = Concatenate(axis=-1)([context, centre])
    merge_instance = Highway(activation=highway_activation)(merge_instance)
    # merge_instance = Dense(units=EMBEDDING_DIM * 2, activation=hidden_activation)(merge_instance)
    # merge_instance = Dropout(dropout_rate)(merge_instance)

    merge_instance = Dense(units=EMBEDDING_DIM, activation=hidden_activation)(merge_instance)
    merge_instance = Dropout(dropout_rate)(merge_instance)

    prediction = Dense(units=1, activation=output_activation)(merge_instance)

    model = Model(inputs=INPUTS, outputs=prediction)

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    if print_model:
        print(model.summary())
        
    return model, embedding_layer