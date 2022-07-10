from flask import Flask,redirect,url_for,request,render_template
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
import os
from keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml

params_path="config/params.yaml"


def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def predict(data):
  config=read_params(params_path)
  MODEL_PATH = config['model_dir']
  print(MODEL_PATH)
  model = load_model(MODEL_PATH)
  onehot_repr=[one_hot(data,5000)] 
  x=pad_sequences(onehot_repr,maxlen=200)
  s=model.predict(x)
  s=np.where(s > 0.5, 1,0)

  if s[0][0]==1:
    return "spam email"
  else:
    return "not spam" 