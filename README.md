
# <a href="url"><img src="https://user-images.githubusercontent.com/61903698/178227001-a84748ea-5fe9-4157-8db1-4a4efc94f694.png" align="left" height="48" width="48" ></a>  Spam Message Classifier



The Aim of the project is to find the text given by the user is spam or not.

## Demo
![spam_gif](https://user-images.githubusercontent.com/61903698/178225864-9e86bda1-0c6e-4665-9a55-7c6c614ebaae.gif)


for live project you can visit https://spamclass.herokuapp.com/
# Installation

### Libraries
* dvc
* dvc[gdrive]
* sklearn
* pandas
* pytest
* tox
* flake8
* flask
* gunicorn
* PyYAML
* tensorflow-cpu
* nltk
* mlflow

to install above libraries please run the command

```bash
     pip install requirements.txt
```

## Deployment

To deploy this project run

```bash
  python src/get_data.py
  python src/load_data.py
  python src/data_preprocessing.py
  python src/split_data.py
  python src/train_evaluate.py 
```

#### Git Commands
```bash
     git remote add origin https://github.com/samrajkodai/samrajkodai.git
     git add .
     git commit -m "your text"
     git branch -M main   
     git push -m origin main

```

## Screenshots

![Screenshot 2022-07-11 134833](https://user-images.githubusercontent.com/61903698/178226366-6878cb78-35c3-4f9d-8d26-5c3e9697836c.jpg)

## Badges

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)


## AI Tools and Framework
* Machine Learning
* Deep Learning
* LSTM
* Natural Language Processing
* Flask
* Mlops
* Dvc
* Git
* Vscode
