import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

# 모델 로드
def load_model_with_custom_objects(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# 새로운 음악 생성
def generate_music(model, seed, length=2048):
    generated = seed
    for _ in range(length):
        # 입력 데이터의 형태를 올바르게 맞춤
        prediction = model.predict(generated[:, -1, :].reshape(1, -1, generated.shape[2]))
        generated = np.concatenate((generated, prediction), axis=1)
    return generated

# 스펙트로그램을 오디오로 변환
def spectrogram_to_audio(S_db, hop_length=512, gain=1.0):
    S = librosa.db_to_amplitude(S_db)
    y = librosa.istft(S, hop_length=hop_length)
    y = y * gain  # 볼륨 조정
    return y

if __name__ == "__main__":
    model_path = "models/music_generator_model.h5"
    model = load_model_with_custom_objects(model_path)

    seed = np.random.rand(1, 1, 2048)  # 초기 시드의 크기를 모델 입력 크기에 맞춤
    generated_music = generate_music(model, seed, length=2048)

    # 마지막 차원을 제거하여 2D 스펙트로그램으로 변환
    generated_music = generated_music[0, :, :]

    # 스펙트로그램을 오디오로 변환
    gain = 5.0  # 볼륨을 조정
    generated_audio = spectrogram_to_audio(generated_music, gain=gain)

    # WAV 파일로 저장
    wav_path = "generated_music.wav"
    sf.write(wav_path, generated_audio, 22050)
