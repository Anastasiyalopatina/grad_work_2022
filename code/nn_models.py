import gensim
import numpy as np
import os
import pickle
from keras.constraints import max_norm
from keras.layers import Dense
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.layers import Embedding
from keras.optimizers import adam_v2
from keras.models import model_from_json

def create_cnn_model(embedding_dim, sequence_length, num_filters,
    num_classes, key_to_indexulary, learning_rate, lang):
    filter_sizes = [3,5,7]
    drop = 0.5
    embedding = None
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding_1d = pre_embedding(embedding_dim = embedding_dim, seq_length = sequence_length,
        input = inputs, voc = key_to_indexulary, lang = lang)
    embedding = Reshape((sequence_length, embedding_dim, 1))(embedding_1d)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),
     padding='valid', kernel_initializer='normal', activation='relu',
      kernel_constraint=max_norm(3))(embedding)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim),
     padding='valid', kernel_initializer='normal', activation='relu',
      kernel_constraint=max_norm(3))(embedding)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),
     padding='valid', kernel_initializer='normal', activation='relu',
      kernel_constraint=max_norm(3))(embedding)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1,1),
     strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
     strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
     strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)

    output2 = Dense(units = num_classes, activation = 'softmax')(dropout)

    model = Model(inputs=inputs, outputs=output2)
    optimizer = adam_v2.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    return model

def pre_embedding(embedding_dim, seq_length, voc, input = None, model = None, lang = 'russian'):

    embed_saved_path =  os.path.join(os.path.dirname(__file__), '../resources',
    'embed_' + str(seq_length) + '_' + lang)
    if os.path.exists(embed_saved_path):
        embed_saved_file = open(embed_saved_path, 'rb')
        embedding_matrix = pickle.load(embed_saved_file)
    else:
        w2v = {}
        if lang == 'russian':
            w2v_english_dir = os.path.join(os.path.dirname(__file__), '../resources',
                'wiki.ru.vec')
        elif lang == 'english':
            w2v_english_dir = os.path.join(os.path.dirname(__file__), '../resources',
                'wiki.en.vec')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_english_dir, binary=False)

        print('Embedding Vocab Size', len(w2v.key_to_index))
        count = 0
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(voc) + 1, embedding_dim))

        for word, i in voc.items():

            if word not in  w2v.key_to_index:
                continue
            embedding_vector = w2v.get_vector(word)

            if embedding_vector is not None:
                count+=1
                embedding_matrix[i] = embedding_vector

        print('Found: ', count, ' words')
        embed_saved_file = open(embed_saved_path, 'wb')
        pickle.dump(embedding_matrix, embed_saved_file)
    
    if input != None:
        embedding =  Embedding(input_dim = len(voc) + 1,
                                output_dim = embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length)(input)
        return embedding
    elif model != None:
        model.add(Embedding(input_dim = len(voc) + 1,
                                output_dim = embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length))


def save_model(model, filename):
    model_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    if not os.path.exists(model_dir_save_path):
        os.makedirs(model_dir_save_path)

    model_json = model.to_json()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../models', filename + '.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../models', filename + '.h5'))    

    print('Model Saved')


def restore_model(filename = 'cnnF-512_dT-csv_bS-32_E-1_sL-200_lR-0.001_L-russian'):
    # file = open(os.path.join(filename, '.json') , 'r')
    file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../models', filename + '.json') , 'r')
    loaded  = file.read()
    file.close()

    model = model_from_json(loaded)
    model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../models', filename + '.h5'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model

