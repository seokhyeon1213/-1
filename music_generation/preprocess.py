import os
import librosa
import numpy as np

# 오디오 파일을 스펙트로그램으로 변환
def audio_to_spectrogram(file_path, n_fft=2048, hop_length=512, n_mels=128):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# 스펙트로그램 크기 고정
def fix_spectrogram_shape(S_db, max_frames=2048):
    if S_db.shape[1] > max_frames:
        return S_db[:, :max_frames]
    else:
        padding = max_frames - S_db.shape[1]
        return np.pad(S_db, ((0, 0), (0, padding)), mode='constant')

# 스펙트로그램 저장
def save_spectrogram(S_db, save_path):
    np.save(save_path, S_db)

# 모든 오디오 파일에 대해 스펙트로그램 생성 및 저장
def process_all_audio_files(audio_dir, output_dir, max_frames=2048):
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".mp3") or file_name.endswith(".wav"):
            file_path = os.path.join(audio_dir, file_name)
            S_db = audio_to_spectrogram(file_path)
            S_db_fixed = fix_spectrogram_shape(S_db, max_frames=max_frames)
            save_path = os.path.join(output_dir, file_name.replace(".mp3", "").replace(".wav", "") + '.npy')
            save_spectrogram(S_db_fixed, save_path)

if __name__ == "__main__":
    audio_dir = "data/audio_files"
    output_dir = "data/spectrograms"
    os.makedirs(output_dir, exist_ok=True)
    process_all_audio_files(audio_dir, output_dir, max_frames=2048)
