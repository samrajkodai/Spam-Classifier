
from flask import Flask,redirect,url_for,request,render_template
import os
from prediction_service import prediction

webapp_root="webapp"
static_dir=os.path.join(webapp_root,"static")
template_dir=os.path.join(webapp_root,"templates")

app=Flask(__name__,static_folder=static_dir,template_folder=template_dir)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
      data=request.get_data()
      print("data",request.form['new_freq'])
      return prediction.predict(request.form['new_freq'])
      
    return None

if __name__=="__main__":
    app.run(host="127.0.0.1",port=7000)