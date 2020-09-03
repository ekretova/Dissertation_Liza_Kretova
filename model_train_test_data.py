import os
import pickle
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from finbert_embedding.embedding import FinbertEmbedding
from tqdm import tqdm

prepocessed_data_directory = 'D:/dis/final-preprocessing-pickle-data'
preprocessed_file_name = 'mda_aux_df_movement.pickle'
model_output_folder = 'D:/dis/finbert-modelling'
training_data_output_folder = 'D:/dis/model-input-data'


#Begin by importing finalised pre-processing pickle file, combining auxiliary
def load_preprocessing_data():
    infile = open(os.path.join(prepocessed_data_directory, preprocessed_file_name), 'rb')
    preprocessed_mdas = pickle.load(infile)
    return preprocessed_mdas


def ttest_training_split(preprocessed_mdas):
    #Create training and test data set based on date values
    #Training set is for all dates prior to 2018-07-10
    #Test set is for all dates after 2018-07-10
    split_date=pd.to_datetime(datetime.date(2018, 7, 10))
    final_test_set_df = preprocessed_mdas[preprocessed_mdas['Released Date:'] >= str(split_date)]
    training_set_df= preprocessed_mdas[preprocessed_mdas['Released Date:'] < str(split_date)]

    final_test_set_df=final_test_set_df.reset_index()
    training_set_df=training_set_df.reset_index()

    return final_test_set_df, training_set_df



def stock_movement_training_test(test_df,training_df):
    #Gather stock price movements for both the training and test dataframes
    training_y = training_df['Stock Price Movements:']
    test_y = test_df['Stock Price Movements:']


    training_y = pd.get_dummies(columns=['Stock Price Movements:'],data=training_y)
    y_test = pd.get_dummies(columns=['Stock Price Movements:'],data=test_y)


    return y_test, training_y


def aux_data_training_test(test_df,training_df,aux_columns):
    #Gather auxiliary data columns from both the training and test dataframes
    final_test_aux_X = test_df[aux_columns]
    training_aux_X = training_df[aux_columns]

    final_test_aux_X = pd.get_dummies(data=final_test_aux_X)
    training_aux_X = pd.get_dummies(data=training_aux_X)

    test_aux = final_test_aux_X.fillna(0)

    return test_aux,training_aux_X

#Do something with processed MD&A texts
def tokenise_training_test_filings(test_df,training_df):
    training_filings  = training_df['Processed Text:']
    test_filings = test_df['Processed Text:']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_filings)
    vocabulary_size = len(tokenizer.word_index) + 1
    max_words =  max(list([len(training_filings.iloc[i]) for i in range(len(training_filings))]))

    training_processed_filings = pad_sequences(sequences = tokenizer.texts_to_sequences(training_filings),maxlen = max_words, padding = 'post')
    test_processed_filings = pad_sequences(sequences = tokenizer.texts_to_sequences(test_filings),maxlen = max_words, padding = 'post')

    return test_processed_filings,training_processed_filings,tokenizer,vocabulary_size,max_words



def split_training_validation_data(tokenised_filings_training,stock_movement_training, aux_training ):
    filings_train, filings_valid, y_train, y_valid, aux_train, aux_valid = train_test_split(tokenised_filings_training,
                                                                                            stock_movement_training, aux_training,
                                                                                            stratify=stock_movement_training,
                                                                                            test_size=0.2,
                                                                                            random_state=20)
    aux_train = aux_train.fillna(0)
    aux_valid = aux_valid.fillna(0)

    return filings_train, filings_valid, y_train, y_valid, aux_train, aux_valid

def initialise_embedding_matrix(vocabulary_size,tokenizer):
    embedding_dimension = 768
    finbert = FinbertEmbedding()
    embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))
    for word, i in tqdm(tokenizer.word_index.items()):
        embedding_vector = finbert.word_vector(word)[0]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    with open(model_output_folder + 'embedding_matrix1.pickle', 'wb') as f:
        pickle.dump(embedding_matrix,f)

def load_embedding_matrix():
    infile1 = open(os.path.join(model_output_folder, 'embedding_matrix1.pickle'), 'rb')
    embedding_matrix = pickle.load(infile1)
    return embedding_matrix

#Split training and test data
def create_train_test_dataset():
    preprocessed_mdas=load_preprocessing_data()

    # Set variable for all columns that are related to auxiliary data
    aux_columns = ['30 Day momentum', '180 Day momentum', '360 Day momentum', 'Volatility']

    test_df, training_df = ttest_training_split(preprocessed_mdas)

    test_returns = test_df[['Return']]
    print(type(test_returns))

    #Get test and training (training+validation) data for stock movements
    test_y, training_y = stock_movement_training_test(test_df, training_df)



    #Get produces tokenised test and training data for processed MD&As
    test_filings, training_tokenised_filings, tokenizer,vocabulary_size,max_words = tokenise_training_test_filings(test_df,training_df)



    ##Get test and training (training+validation) data for auxiliary data
    test_aux, aux_training = aux_data_training_test(test_df,training_df,aux_columns)


    train_filings, valid_filings, train_y, valid_y, train_aux, valid_aux = split_training_validation_data(training_tokenised_filings, training_y, aux_training)

    #Save training and test data sets as future inputs for model
    with open(os.path.join(training_data_output_folder, 'test-train-data.pickle'), 'wb') as f:
        pickle.dump([test_returns,test_y, train_y, valid_y,test_filings,train_filings,valid_filings,test_aux,train_aux,valid_aux,tokenizer,vocabulary_size,max_words], f)

    #save aux data
    with open(os.path.join(training_data_output_folder, 'aux-test-train-data.pickle'), 'wb') as f:
        pickle.dump([test_aux, train_aux,valid_aux], f)

    #save filings
    with open(os.path.join(training_data_output_folder, 'mda-test-train-data.pickle'), 'wb') as f:
        pickle.dump([test_filings, train_filings, valid_filings], f)

    #save movements
    with open(os.path.join(training_data_output_folder, 'movement-test-train-data.pickle'), 'wb') as f:
        pickle.dump([test_y, train_y, valid_y], f)

    #save embedding details
    with open(os.path.join(training_data_output_folder, 'embedding-data.pickle'), 'wb') as f:
        pickle.dump([tokenizer, vocabulary_size, max_words], f)

    #save returns
    with open(os.path.join(training_data_output_folder, 'return-test-data.pickle'), 'wb') as f:
        pickle.dump(test_returns, f)


def load_aux_dataset():
    infile = open(os.path.join(training_data_output_folder, 'aux-test-train-data.pickle'), 'rb')
    test_aux,train_aux,valid_aux = pickle.load(infile)
    return test_aux,train_aux,valid_aux

def load_mda_dataset():
    infile = open(os.path.join(training_data_output_folder, 'mda-test-train-data.pickle'), 'rb')
    test_filings, train_filings, valid_filings = pickle.load(infile)
    return test_filings, train_filings, valid_filings

def load_movement_dataset():
    infile = open(os.path.join(training_data_output_folder, 'movement-test-train-data.pickle'), 'rb')
    test_y, train_y, valid_y = pickle.load(infile)
    return test_y, train_y, valid_y

def load_embedding_dataset():
    infile = open(os.path.join(training_data_output_folder, 'embedding-data.pickle'), 'rb')
    tokenizer, vocabulary_size, max_words = pickle.load(infile)
    return tokenizer, vocabulary_size, max_words

def load_returns_dataset():
    infile = open(os.path.join(training_data_output_folder, 'return-test-data.pickle'), 'rb')
    test_returns = pickle.load(infile)
    return test_returns

create_train_test_dataset()


