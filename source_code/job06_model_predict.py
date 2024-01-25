import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from konlpy.tag import Okt  # 한국어 자연어 처리(형태소 분리기) 패키지

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle  # 파이썬 데이터형을 그대로 저장 하는 파일 저장 패키지

from tensorflow.keras.models import load_model

df = pd.read_csv("../crawling_data/naver_headline_news_20240124.csv")
print(df.head())
print(df.info())

X = df["titles"]
Y = df["category"]

with open("../models/label_encoder.pickle", "rb") as file:
    label_encoder = pickle.load(file)

# labeled_y = label_encoder.transform(Y)  # 기존에 있는 classes 정보로 라벨링을 하기 위해선 transform만 사용, fit_transform을 하게 되면 새로운 라벨이 생성되기 때문에
label = label_encoder.classes_  # 기존에 만들었던 레이블을 불러옴

print(label)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem = True)

stopwords = pd.read_csv("../stopwords.csv")

for i in range(len(X)):
    words = []
    for j in range(len(X[i])):
        if len(X[i][j]) > 1:
            if X[i][j] not in list(stopwords["stopword"]):
                words.append(X[i][j])

    X[i] = " ".join(words)

with open("../models/news_token.pickle", "rb") as file:
    token = pickle.load(file)

tokened_x = token.texts_to_sequences(X)

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 27:
        tokened_x[i] = tokened_x[i][:27]

print(tokened_x)

x_pad = pad_sequences(tokened_x, 27)

model = load_model("../models/news_category_classification_model_0.7162036895751953.h5")

preds = model.predict(x_pad)

predicts = []

for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0  # 제일 큰 값을 0으로 만듦
    second = label[np.argmax(pred)]  # 제일 큰 값이 0이 되었기 때문에 두번째 큰값을 찾음
    predicts.append([most, second])

df["predict"] = predicts

print(df)


df["OX"] = 0

for i in range(len(df)):
    if df.loc[i, "category"] in df.loc[i, "predict"]:  # 리스트 안에 있는지 판단 
        df.loc[i, "OX"] = "O"
    else:
        df.loc[i, "OX"] = "X"

print(df["OX"].value_counts())

print(df["OX"].value_counts() / len(df))  # 정확도가 떨어지는 이유는 새로운 뉴스가 나왔기 때문


