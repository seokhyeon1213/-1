import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Bidirectional, Input, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

# 데이터 로드
def load_data(data_dir):
    data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(data_dir, file_name)
            data.append(np.load(file_path))
    data = np.array(data)
    return data

# 모델 정의
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(input_shape[0] * input_shape[1], activation='sigmoid'))
    model.add(Reshape(input_shape))
    return model

if __name__ == "__main__":
    data_dir = "data/spectrograms"
    data = load_data(data_dir)
    
    # 데이터 형태 확인
    if len(data) == 0:
        raise ValueError("No data found. Ensure that the spectrogram files are correctly saved in the data/spectrograms directory.")
    if len(data.shape) != 3:
        raise ValueError(f"Unexpected data shape: {data.shape}. Expected a 3D array.")
    
    # 데이터 형태 출력
    print(f"Data shape: {data.shape}")
    
    input_shape = (data.shape[1], data.shape[2])
    model = build_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(data, data, epochs=100, batch_size=4, validation_split=0.2)
    model.save("models/music_generator_model.h5")
