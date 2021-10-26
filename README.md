# CoronavirusTweetClassification

1. [Introduction](#Introduction)
2. [How to run project](#How-to-run-project)
3. [Contributors](#Contributors)

## Introduction

Sentiment analysis (also known as opinion mining or emotion AI) is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.
The aim of the project is the classification of the texts contained in the tweets relating to Covid-19.

## How to run project

1. Clone this [project](https://github.com/martinasalis/CoronavirusTweetClassification)
```bash
git clone https://github.com/martinasalis/CoronavirusTweetClassification.git
```

2. Install requirements
```bash
cd CoronavirusTweetClassification
pip3 install -r requirements.txt
```
3. Download Google Word2Vec pre-trained model from this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) and unzip the file in ```dataset/``` folder.

4. The code is currently training the neural network. You can execute other pieces of code by changing the flags found from line 180 to line 184 of the ```main.py``` file. To run the project enter the following command:
```bash
python3 main.py
```

## Contributors
[Martina Salis](https://github.com/martinasalis) <br/>
[Luca Grassi](https://github.com/Luca14797)
