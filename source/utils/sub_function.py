
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class HF_STFTDataset(Dataset):
    def __init__(self, hf_dataset, n_fft=512, win_length=320, hop_length=160, duration=10):
        self.dataset = hf_dataset
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.max_len = int(16000 * duration)
        self.window = torch.hann_window(self.win_length)

    def __len__(self): return len(self.dataset)

    def process_len(self, wav_array):
        wav = torch.from_numpy(wav_array).float()
        if wav.ndim == 1: wav = wav.unsqueeze(0)
        if wav.shape[1] > self.max_len:
            start = random.randint(0, wav.shape[1] - self.max_len)
            return wav[:, start:start + self.max_len], start
        return torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[1])), 0

    def get_stft(self, wav):
        stft_complex = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                                  win_length=self.win_length, window=self.window,
                                  center=True, return_complex=True)
        return torch.view_as_real(stft_complex).squeeze(0)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        mic_wav, start = self.process_len(item['mic']['array'])
        ref_wav, _ = self.process_len(item['ref']['array'])
        clean_wav, _ = self.process_len(item['clean']['array'])
        
        # Lấy nhãn VAD từ dataset (dạng List)
        vad_full = torch.tensor(item['vad_label'], dtype=torch.long)
        

        num_frames = self.get_stft(mic_wav).shape[1]
        start_frame = start // self.hop_length
        vad_label = vad_full[start_frame : start_frame + num_frames]
        
        if vad_label.shape[0] < num_frames:
            vad_label = torch.nn.functional.pad(vad_label, (0, num_frames - vad_label.shape[0]))
        elif vad_label.shape[0] > num_frames:
            vad_label = vad_label[:num_frames]

        return self.get_stft(mic_wav), self.get_stft(ref_wav), self.get_stft(clean_wav), vad_label

