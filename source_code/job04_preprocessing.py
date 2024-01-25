import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from konlpy.tag import Okt  # 한국어 자연어 처리(형태소 분리기) 패키지

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle  # 파이썬 데이터형을 그대로 저장 하는 파일 저장 패키지

# 자연어 처리에 있어 문자를 숫자로 변환 하는 과정 : Tokenize
# tokenize 할 때 형태소 단위로 진행 해야함

# 파일 불러오기
df = pd.read_csv("../datasets/naver_news_titles_20240125.csv")
# print(df.head())
# print(df.info())

X = df["titles"]
Y = df["category"]  # 6개의 카테고리 onehot encoding

# y 전처리
label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y)  # labeling을 해야 classes 값이 생김

# print(labeled_y[:3])

label = label_encoder.classes_  # ['Culture' 'Economics' 'IT' 'Politics' 'Social' 'World'] 이 순서로 라벨링을 함

# print(label)

with open("../models/label_encoder.pickle", "wb") as file:  # wb : binary로 저장
    pickle.dump(label_encoder, file)  # 라벨 인코더 정보를 저장
    
onehot_y = to_categorical(labeled_y)

# print(onehot_y[:3])

# x 전처리 : 자연어 처리

okt = Okt()  # 형태소 분리

# print(X[1:5])

# 모든 문장에 대해 형태소 분리
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem = True)  # stem : 원형으로 변형
                                          # stem이 없을 경우 그냥 형태소로 잘라주만 함
    if i % 1000 == 0:
        print(i)
# print(X[:5])

# stop word(불용어 제거)
stopwords = pd.read_csv("../stopwords.csv", index_col = 0)

for j in range(len(X)):
    words = []

    for i in range(len(X[j])):
        if len(X[j][i]) > 1:  # 형태소의 길이가 1보다 길며 
            if X[j][i] not in list(stopwords["stopword"]):  # 형태소가 stopword에 포함 되어 있지 않으면
                words.append(X[j][i])

    X[j] = " ".join(words)  # 하나의 문장으로 만들음

# print(X[:5])

token = Tokenizer()  # 형태소를 토큰(label)화
token.fit_on_texts(X)  # 형태소를 token으로 정렬
# 새로운 문장에 대해 fit_on_texts를 하게 되면 인덱스가 변할수도 있기 때문에 token을 저장해야 함

tokened_x = token.texts_to_sequences(X)  # 각 문장을 토큰으로 변형 하여 숫자 리스트로 변형
wordsize = len(token.word_index) + 1  # 인덱스가 1부터 시작함
# print(tokened_x)
print(f"형태소의 개수 {wordsize}")

with open("../models/news_token.pickle", "wb") as file:
    pickle.dump(token, file)  # token(단어의 인덱스)를 저장

# 가장 긴 문장에 사이즈를 맞추며, 앞쪽에 0으로 채움
max = 0

for i in tokened_x:
    if max < len(i):
        max = len(i)

print(f"가장 긴 문장의 길이 : {max}")

x_pad = pad_sequences(tokened_x, max)  # 앞쪽에 자동으로 0으로 채워줌
# 0 : 길이를 맞추거나, 학습 당시에 사용한 token이 아닌 경우에 0을 사용

print(x_pad)

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size = 0.2
)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = np.array((X_train, X_test, Y_train, Y_test), dtype = object)
np.save("../datasets/news_data_max_{}_wordsize_{}.npy".format(max, wordsize), xy)
