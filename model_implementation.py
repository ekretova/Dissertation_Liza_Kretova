from keras import optimizers
from keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN, LeakyReLU
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import concatenate as lconcat
from keras.models import Sequential, Model, load_model

from model_ouput_analysis import *
from model_train_test_data import *

model_output_folder = 'D:/dis/finbert-modelling'

test_y, train_y, valid_y = load_movement_dataset()
test_filings,train_filings,valid_filings = load_mda_dataset()
test_aux,train_aux,valid_aux = load_aux_dataset()
tokenizer,vocabulary_size,max_words=load_embedding_dataset()
test_returns = load_returns_dataset()



embedding_matrix = load_embedding_matrix()

#First model uses only aux data, using CNN initially, together with LSTM.
def mlp_auxiliary(output_classes,aux_feats_nb = 4,lr =0.001,weight_decay = 0.01):
    model = Sequential()
    aux_input = Input(shape=(aux_feats_nb,), name='aux_input')
    aux_x = Reshape((aux_feats_nb, 1), input_shape=(aux_feats_nb,))(aux_input)
    aux_x = SimpleRNN(2, return_sequences=True, activation='relu')(aux_x)

    aux_x=Dense(50, input_dim=aux_feats_nb)(aux_x)
    aux_x=LeakyReLU(alpha=0.1)(aux_x)
    aux_x=Dense(50, input_dim=aux_feats_nb)(aux_x)
    aux_x=LeakyReLU(alpha=0.1)(aux_x)

    aux_x=Dropout(0.1)(aux_x)
    aux_x = Flatten()(aux_x)


    main_output = Dense(output_classes, activation='softmax', name='main_output')(aux_x)
    model = Model(inputs=[aux_input], outputs=[main_output],name="mlp")
    adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False,decay = weight_decay)

    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
    print(model.summary())
    return model

aux_columns = ['30 Day momentum','180 Day momentum','360 Day momentum','Volatility']

#mlp_auxiliary = mlp_auxiliary(3,lr = 0.001)

#mlp_auxiliary.fit(train_aux[aux_columns],train_y,batch_size=16,epochs=2,verbose=1,validation_data = (valid_aux[aux_columns],valid_y))

#mlp_auxiliary.save('auxiliary_rnn1.h5')


def load_aux_model_predict(model_filename):
    model = load_model(model_filename)
    model_prediction = model.predict(test_aux[aux_columns],batch_size = 16)

    model_accuracy(model_prediction, test_y,test_returns)

#load_model_predict('auxiliary_rnn1.h5')

def load_embedding_aux_model_predict(model_filename):
    model = load_model(model_filename)
    model_prediction = model.predict([test_filings,test_aux[aux_columns]],batch_size = 16)

    model_accuracy(model_prediction, test_y,test_returns)

#cnn with LSTM on embedding- concat with results from rnn & dense on aux
def mlp_embedding_aux(output_classes,max_words,aux_feats_nb = 4,lr =0.001,weight_decay = 0.01, dropout_rate = 0.1, embedding_matrix = embedding_matrix,vocab_size = vocabulary_size,embedding_dim = 768):
    main_input= Input(shape=(max_words,),name='doc_input')
    main = Embedding(input_dim = vocab_size,
                                output_dim = embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_words,
                                trainable=False)(main_input)

    x = Dense(50)(main)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(50)(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)

    aux_input = Input(shape=(aux_feats_nb,), name='aux_input')
    aux_x = Reshape((aux_feats_nb, 1), input_shape=(aux_feats_nb,))(aux_input)
    aux_x = SimpleRNN(5, return_sequences=True, activation='relu')(aux_x)
    aux_x=Dense(50, input_dim=aux_feats_nb)(aux_x)
    aux_x=LeakyReLU(alpha=0.1)(aux_x)
    aux_x=Dense(50, input_dim=aux_feats_nb)(aux_x)
    aux_x=LeakyReLU(alpha=0.1)(aux_x)
    aux_x = Flatten()(aux_x)

    x = lconcat([x,aux_x])
    x = Dropout(dropout_rate)(x)

    main_output = Dense(output_classes, activation='softmax', name='main_output')(x)
    model = Model(inputs=[main_input,aux_input], outputs=[main_output],name="mlp")
    adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False,decay = weight_decay)

    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
    print(model.summary())
    return model

mlp_2 = mlp_embedding_aux(3,max_words,lr = 0.001)
mlp_2.fit([train_filings,train_aux[aux_columns]],train_y,batch_size=16,epochs=2,verbose=1,validation_data = ([valid_filings,valid_aux[aux_columns]],valid_y))

mlp_2.save('aux_embedding_model.h5')

load_embedding_aux_model_predict('aux_embedding_model.h5')
