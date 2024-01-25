import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load("../datasets/news_data_max_27_wordsize_11938.npy",
                                           allow_pickle = True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential(
    [
        Embedding(11938, 300, input_length = 27),  
        # 자연어의 의미를 학습하는 레이어 input_dim : wordsize, input_length : max. output_dim : 축소 차원수
        Conv1D(32, kernel_size = 5, padding = "same", activation = "relu"),
        MaxPooling1D(pool_size = 1),  # 사이즈가 1이기 때문에 지금은 큰 의미 없음, conv-maxpool은 세트
        LSTM(128, activation = "tanh", return_sequences = True),
        # return_sequence : LSTM은 연속적인 데이터를 입력으로 받기 때문에 셀의 결과 값을 이어주어 연속 데이터로 만들어 줌, default : false
        Dropout(0.3),
        LSTM(62, activation = "tanh", return_sequences = True),
        Dropout(0.3),
        LSTM(64, activation = "tanh"),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation = "relu"),
        Dense(6, activation = "softmax")
    ]
)

model.summary()

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

fit_hist = model.fit(X_train, Y_train, batch_size = 128, epochs = 10, validation_data = (X_test, Y_test))

model.save("../models/news_category_classification_model_{}.h5".format(fit_hist.history["val_accuracy"][-1]))

plt.plot(fit_hist.history["val_accuracy"], label = "validation accuracy")
plt.plot(fit_hist.history["accuracy"], label = "accuracy")

plt.legend()
plt.show()
