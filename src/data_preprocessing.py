from distutils.command.config import config
import imp
import os
from django import conf 
import pandas as pd
from pyparsing import Word, col
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
import argparse
from get_data import read_params,get_data
import re
from sklearn.feature_extraction.text import CountVectorizer
from numpy import savetxt
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocessing(config_path):
    global corpus
    ps=WordNetLemmatizer()
    config=read_params(config_path)

    main_col=config['base']['main_col']

    df=get_data(config_path)

    for i in df.columns:
        if df[i].isnull().sum()>0:
            print(i,df[i].isnull().sum())
            df[i].fillna("0",inplace=True)

    corpus=[]
    ps=WordNetLemmatizer()
    for i in range(0, len(df['email'])):
        print(i)
        review = re.sub('[^a-zA-Z]', ' ', df['email'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    x=corpus

    voc_size=config['dl_params']['voc_size']
    sent_length=config['dl_params']['sent_length']
    padding=config['dl_params']['padding']
    embed_features=config['dl_params']['embedding_vector_features']

    onehot_rep=[one_hot(word,voc_size) for word in corpus]
    embed_docs=pad_sequences(onehot_rep,maxlen=sent_length,padding=padding)




    x=np.array(embed_docs)
    y=df.iloc[:,-1] 

    print(x)

    x_csv=config['load_data']['x_csv']
    y_csv=config['load_data']['y_csv']
    raw_data_path=config["load_data"]["raw_dataset_csv"]
    x=np.save(x_csv, x)
    y=np.save(y_csv, y)
    
    







if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config/params.yaml")
    parsed_orgs=args.parse_args()
    data=preprocessing(config_path=parsed_orgs.config)
