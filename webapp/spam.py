from distutils.log import error
from unittest import result
from flask import Flask,redirect,url_for,request,render_template
import numpy as np
from keras.applications.vgg16 import preprocess_input
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
import os
from keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

params_path="config/params.yaml"
webapp_root="webapp"

static_dir=os.path.join(webapp_root,"static")
template_dir=os.path.join(webapp_root,"templates")

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app=Flask(__name__)



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

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
      data=request.get_data()
      print("data",request.form['new_freq'])
      return predict(request.form['new_freq'])
      
    return None

if __name__=="__main__":
    app.run(host="127.0.0.1",port=7000)