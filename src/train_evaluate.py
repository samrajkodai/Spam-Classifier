import os
from pickletools import optimize
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json
from numpy import load
from get_data import read_params,get_data
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

def evaluation(x,y):
    accuracy=accuracy_score(x,y)
    matrix=confusion_matrix(x,y)

    print("accuracy : ",accuracy,'\n'"matrix : ",matrix)
    return accuracy,matrix

def train_and_evaluate(config_path):
    config = read_params(config_path)
    x_test_data_path = config["split_data"]["x_test_path"]
    x_train_data_path = config["split_data"]["x_train_path"]
    y_test_data_path = config["split_data"]["y_test_path"]
    y_train_data_path = config["split_data"]["y_train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    voc_size=config['dl_params']['voc_size']
    sent_length=config['dl_params']['sent_length']
    padding=config['dl_params']['padding']
    embed_features=config['dl_params']['embedding_vector_features']
    activation=config['dl_params']['activation']
    optimizer=config['dl_params']['optimizer']
    loss=config['dl_params']['loss']
    metrics=config['dl_params']['metrics']
    epochs=config['dl_params']['epochs']
    batch_size=config['dl_params']['batch_size']
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    x_train= load(x_train_data_path,allow_pickle=True)
    x_test=load(x_test_data_path,allow_pickle=True)
    y_train = load(y_train_data_path,allow_pickle=True)
    y_test=load(y_test_data_path,allow_pickle=True)

    model=Sequential()
    model.add(Embedding(voc_size,embed_features,input_length=sent_length))
    model.add(LSTM(100))
    model.add(Dense(units=1,activation=activation))
    model.compile(loss=loss,optimizer=optimizer,metrics=[metrics])

    print(model.summary())

    ### Finally Training
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size)    

    y_pred=model.predict(x_test)  
    print(y_pred)
    (accuracy,matrix)=evaluation(y_test,y_pred.round())



    with open(scores_file, "w") as f:
        scores = {
            "accuracy score": accuracy
        }
        json.dump(scores, f, indent=4)

    


    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir,"spam classifier.h5"))
    

       
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config/params.yaml")
    parsed_args=args.parse_args()
    data=train_and_evaluate(config_path=parsed_args.config)