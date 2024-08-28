from tensorflow.keras.layers import *
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import os
# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
import tensorflow as tf

import random as rn
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(150)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(150)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from tensorflow.python.keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.random.set_seed(150)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
###### Try changing these parameters to fine tune model ###########

num_lstm = 128
num_dense = 64
num_conv = 128
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
act = 'relu'
####################################################################

def my_lstm(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)

	
	encoding = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm) (embedded_seq)
		
	x = Dropout(rate_drop_dense)(encoding)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)
	
def my_gru(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)
	
	encoding = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm) (embedded_seq)
	
	x = Dropout(rate_drop_dense)(encoding)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)

def my_rnn(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)
	
	encoding = SimpleRNN(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm) (embedded_seq)
	
	x = Dropout(rate_drop_dense)(encoding)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)




def my_stacked_gru(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)
	
	encoding = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True) (embedded_seq)
	encoding = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm) (encoding)

	x = Dropout(rate_drop_dense)(encoding)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)

def my_stacked_lstm(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)

	
	encoding = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True) (embedded_seq)
	encoding = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm) (encoding)
		
	x = Dropout(rate_drop_dense)(encoding)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)

def my_bidirectional_lstm(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)

	
	encoding = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)) (embedded_seq)
		
	x = Dropout(rate_drop_dense)(encoding)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)

def my_cnn_unigram(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)

	
	encoding  = Conv1D(filters=num_conv, kernel_size=1, padding='same', activation='relu') (embedded_seq)
	globAvg = GlobalAveragePooling1D()(encoding)
	globMax = GlobalMaxPooling1D()(encoding)
	merged = concatenate([globAvg, globMax])

		
	x = Dropout(rate_drop_dense)(merged)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)

def my_cnn_bigram(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)

	
	encoding  = Conv1D(filters=num_conv, kernel_size=2, padding='same', activation='relu') (embedded_seq)
	globAvg = GlobalAveragePooling1D()(encoding)
	globMax = GlobalMaxPooling1D()(encoding)
	merged = concatenate([globAvg, globMax])

		
	x = Dropout(rate_drop_dense)(merged)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)

def my_cnn_unibigram(nb_words, embedding_matrix, embedding_dim, nbr_seq, nbr_classes):
	embedding_layer = Embedding(nb_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=nbr_seq,
                            trainable=False)


	sent_in = Input(shape=(nbr_seq,), dtype='int32')
	embedded_seq = embedding_layer(sent_in)

	
	encoding1  = Conv1D(filters=num_conv, kernel_size=1, padding='same', activation='relu') (embedded_seq)
	encoding2  = Conv1D(filters=num_conv, kernel_size=2, padding='same', activation='relu') (embedded_seq)
	globAvg1 = GlobalAveragePooling1D()(encoding1)
	globMax1 = GlobalMaxPooling1D()(encoding1)
	globAvg2 = GlobalAveragePooling1D()(encoding2)
	globMax2 = GlobalMaxPooling1D()(encoding2)
	merged = concatenate([globAvg1, globMax1, globAvg2, globMax2])

		
	x = Dropout(rate_drop_dense)(merged)
	x = BatchNormalization()(x)
	x = Dense(num_dense, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)

	x = Dense(32, activation=act)(x)
	x = Dropout(rate_drop_dense)(x)
	x = BatchNormalization()(x)
	x_pred = Dense(nbr_classes, activation='softmax')(x)
	model = Model(inputs=sent_in, outputs=x_pred)
	return(model)