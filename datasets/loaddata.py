import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def download_and_extract_audio(dataset_name, output_dir):
    folders = ['mic', 'clean', 'ref']
    for folder in folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)


    print(f"Đang tải dataset: {dataset_name}...")
    ds = load_dataset(dataset_name, split='train') 

    print("Bắt đầu giải nén và lưu file âm thanh...")
    
    for i, row in enumerate(tqdm(ds)):
        filename = f"{i:06d}.wav"

        
        for folder in folders:
            audio_data = row[folder]['array']
            sr = row[folder]['sampling_rate']
            
            save_path = os.path.join(output_dir, folder, filename)
            sf.write(save_path, audio_data, sr)

    print(f"Hoàn thành! Dữ liệu đã được lưu tại: {output_dir}")

DATASET_NAME = 'PandaLT/microsoft-AEC-dataset' 
OUTPUT_FOLDER = 'processed_dataset'

if __name__ == "__main__":
    download_and_extract_audio(DATASET_NAME, OUTPUT_FOLDER)