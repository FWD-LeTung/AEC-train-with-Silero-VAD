import torch
import numpy as np
import os
from tqdm import tqdm

input_folder = '/media/disk_360GB/00_datasets/material/processed_dataset/clean'
output_folder = '/media/disk_360GB/00_datasets/material/processed_dataset/labels_10ms'
os.makedirs(output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
model.to(device)
(get_speech_timestamps, _, read_audio, _, _) = utils

def process_batch():
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.wav', '.flac'))]
    print(f"Tìm thấy {len(audio_files)} file. Bắt đầu xử lý...")

    for file_name in tqdm(audio_files):
        try:
            audio_path = os.path.join(input_folder, file_name)
            
            # Đọc audio (sr=16000)
            sampling_rate = 16000
            wav = read_audio(audio_path, sampling_rate=sampling_rate).to(device)
            
            # Lấy timestamps
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, threshold=0.5)
            
            # Tính toán frames
            samples_per_10ms = int(sampling_rate * 0.01) # 160 samples
            total_samples = len(wav)
            total_frames = int(np.ceil(total_samples / samples_per_10ms))
            
            labels = np.zeros(total_frames, dtype=np.int8)
            
            for ts in speech_timestamps:
                start_frame = int(ts['start'] / samples_per_10ms)
                end_frame = int(ts['end'] / samples_per_10ms)
                # Gán nhãn 1 cho các frame thuộc đoạn speech
                labels[start_frame:end_frame + 1] = 1
            
            # Tạo tên file output (ví dụ: 000000.wav -> 000000.npy)
            output_filename = os.path.splitext(file_name)[0] + '.npy'
            output_path = os.path.join(output_folder, output_filename)
            
            # Lưu file numpy
            np.save(output_path, labels)
            
        except Exception as e:
            print(f"\n Lỗi tại file {file_name}: {e}")

if __name__ == "__main__":
    process_batch()
    print(f" Hoàn thành! Nhãn đã được lưu tại: {output_folder}")