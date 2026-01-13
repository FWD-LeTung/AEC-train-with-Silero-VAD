import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH VÀ SIÊU THAM SỐ
# ==========================================
CONFIG = {
    "mic_dir": "/media/disk_360GB/00_datasets/material/processed_dataset/mic",
    "ref_dir": "/media/disk_360GB/00_datasets/material/processed_dataset/ref",
    "clean_dir": "/media/disk_360GB/00_datasets/material/processed_dataset/clean",
    "label_dir": "/media/disk_360GB/00_datasets/material/processed_dataset/vadlabel",
    "sr": 16000,
    "n_fft": 512,
    "win_length": 320,
    "hop_length": 160,
    "d_model": 128,
    "batch_size": 16,
    "lr": 2e-4,
    "epochs": 10,
    "beta": 0.5,           # Trọng số VAD Loss
    "val_samples": 500,    # 500 mẫu đầu tiên cho Val
    "val_step": 100,       # Step chạy Validation
    "log_step": 10,        # Step ghi log WandB
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH AEC (CONFORMER)
# ==========================================
class CasualMHSA(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
    def forward(self, x):
        B, T, D = x.shape
        attn_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        x_out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return x_out

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size-1)//2, groups=d_model)
        self.batch_norm = nn.GroupNorm(num_groups=1, num_channels=d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.pointwise_conv2(self.activation(self.batch_norm(self.depthwise_conv(self.glu(self.pointwise_conv1(x)))))))
        return x.transpose(1, 2)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_model*expansion_factor)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_model*expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout2(self.layer2(self.dropout1(self.activation(self.layer1(x)))))

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.self_attn = CasualMHSA(d_model, n_head, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model); self.norm4 = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.norm1(x))
        x = x + self.conv_module(self.norm2(x))
        x = x + self.self_attn(self.norm3(x))
        x = x + 0.5 * self.ffn2(self.norm4(x))
        return self.final_norm(x)

class AECModel(nn.Module):
    def __init__(self, d_model=128, n_fft=512, n_head=8, num_layers=4):
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        input_dim = self.n_freq * 4 
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([ConformerBlock(d_model, n_head) for _ in range(num_layers)])
        self.mask_proj = nn.Linear(d_model, self.n_freq * 2)

    def forward(self, mic_stft, ref_stft):
        B, F, T, C = mic_stft.shape
        mic_flat = mic_stft.permute(0, 2, 1, 3).reshape(B, T, F * 2)
        ref_flat = ref_stft.permute(0, 2, 1, 3).reshape(B, T, F * 2)
        x = torch.cat([mic_flat, ref_flat], dim=2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        mask = self.mask_proj(x).view(B, T, F, 2).permute(0, 2, 1, 3)
        mic_real, mic_imag = mic_stft[..., 0], mic_stft[..., 1]
        mask_real, mask_imag = mask[..., 0], mask[..., 1]
        est_real = mic_real * mask_real - mic_imag * mask_imag
        est_imag = mic_real * mask_imag + mic_imag * mask_real
        return torch.stack([est_real, est_imag], dim=-1)

# ==========================================
# 3. DATASET HỖ TRỢ CHIA TRAIN/VAL
# ==========================================
class AECVADDataset(Dataset):
    def __init__(self, file_list, mic_dir, ref_dir, clean_dir, label_dir, config):
        self.cfg = config
        self.file_names = file_list
        self.mic_dir = mic_dir
        self.ref_dir = ref_dir
        self.clean_dir = clean_dir
        self.label_dir = label_dir
        self.target_samples = int(config['sr'] * 10.0) # Cố định 10 giây

    def __len__(self):
        return len(self.file_names)

    def _load_and_fix_length(self, path):
        wav, _ = torchaudio.load(path)
        wav = wav[0]
        if wav.shape[0] > self.target_samples:
            wav = wav[:self.target_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.target_samples - wav.shape[0]))
        return wav

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        mic = self._load_and_fix_length(os.path.join(self.mic_dir, fname))
        ref = self._load_and_fix_length(os.path.join(self.ref_dir, fname))
        clean = self._load_and_fix_length(os.path.join(self.clean_dir, fname))
        
        label_path = os.path.join(self.label_dir, fname.replace('.wav', '.npy'))
        vad_label = torch.from_numpy(np.load(label_path)).float()

        # Không truyền window để giống code cũ (dùng rectangular window)
        stft_params = {
            'n_fft': self.cfg['n_fft'], 
            'hop_length': self.cfg['hop_length'], 
            'win_length': self.cfg['win_length'], 
            'center': True, 
            'return_complex': True
        }
        
        mic_stft = torch.view_as_real(torch.stft(mic, **stft_params))
        ref_stft = torch.view_as_real(torch.stft(ref, **stft_params))
        clean_stft = torch.view_as_real(torch.stft(clean, **stft_params))
        
        num_frames = mic_stft.shape[1] 
        if vad_label.shape[0] > num_frames:
            vad_label = vad_label[:num_frames]
        else:
            vad_label = torch.nn.functional.pad(vad_label, (0, num_frames - vad_label.shape[0]))

        return mic_stft, ref_stft, clean_stft, vad_label

# ==========================================
# 4. HÀM LOSS VÀ DIFFERENTIABLE VAD
# ==========================================
def get_estimated_vad_probs(vad_model, est_wav, sr=16000):
    if hasattr(vad_model, 'reset_states'):
        vad_model.reset_states()
    est_wav = torch.nan_to_num(est_wav, nan=0.0)
    # Peak normalize
    abs_max = torch.max(torch.abs(est_wav), dim=1, keepdim=True)[0] + 1e-7
    est_wav = est_wav / abs_max

    window_size = 512
    hop_size = 160
    est_wav_padded = torch.nn.functional.pad(est_wav, (window_size//2, window_size//2))
    chunks = est_wav_padded.unfold(1, window_size, hop_size)
    b, n_frames, w = chunks.shape
    
    probs = vad_model(chunks.reshape(-1, w), sr)
    # Clamp quan trọng để tránh crash CUDA BCE Loss
    return torch.clamp(probs.reshape(b, n_frames), min=1e-7, max=1-1e-7)

def compute_loss(est_stft, clean_stft, vad_gt, vad_model, cfg):
    est_c = torch.view_as_complex(est_stft)
    clean_c = torch.view_as_complex(clean_stft)
    
    loss_aec = torch.nn.functional.l1_loss(torch.abs(est_c), torch.abs(clean_c)) + \
               torch.nn.functional.l1_loss(est_stft, clean_stft)
    
    # iSTFT không dùng window để giống code cũ
    est_wav = torch.istft(est_c, 
                          n_fft=cfg['n_fft'], 
                          hop_length=cfg['hop_length'], 
                          win_length=cfg['win_length'], 
                          center=True)
    
    est_vad_probs = get_estimated_vad_probs(vad_model, est_wav, cfg['sr'])
    min_f = min(est_vad_probs.shape[1], vad_gt.shape[1])
    loss_vad = torch.nn.functional.binary_cross_entropy(est_vad_probs[:, :min_f], vad_gt[:, :min_f])
    
    return loss_aec, loss_vad

# ==========================================
# 5. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def train():
    device = CONFIG['device']
    wandb.init(project="AEC_VAD_MultiTask_Fix", config=CONFIG)

    all_files = sorted([f for f in os.listdir(CONFIG['mic_dir']) if f.endswith('.wav')])
    val_files = all_files[:CONFIG['val_samples']]
    train_files = all_files[CONFIG['val_samples']:]
    print(f"Dataset Split: Train={len(train_files)}, Val={len(val_files)}")

    train_ds = AECVADDataset(train_files, CONFIG['mic_dir'], CONFIG['ref_dir'], CONFIG['clean_dir'], CONFIG['label_dir'], CONFIG)
    val_ds = AECVADDataset(val_files, CONFIG['mic_dir'], CONFIG['ref_dir'], CONFIG['clean_dir'], CONFIG['label_dir'], CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = AECModel(d_model=CONFIG['d_model']).to(device)
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    vad_model.to(device).eval()
    for p in vad_model.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    global_step = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for mic, ref, clean, vad_gt in pbar:
            mic, ref, clean, vad_gt = mic.to(device), ref.to(device), clean.to(device), vad_gt.to(device)
            
            est_stft = model(mic, ref)
            l_aec, l_vad = compute_loss(est_stft, clean, vad_gt, vad_model, CONFIG)
            total_loss = l_aec + CONFIG['beta'] * l_vad
            
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            total_loss = total_loss.detach()
            optimizer.step()
            
            global_step += 1
            if global_step % CONFIG['log_step'] == 0:
                wandb.log({"train/aec_loss": l_aec.item(), "train/vad_loss": l_vad.item()}, step=global_step)

            if global_step % CONFIG['val_step'] == 0:
                model.eval()
                with torch.no_grad():
                    # Chỉ lấy 1 batch để validate cho nhanh
                    v_mic, v_ref, v_clean, v_vad_gt = next(iter(val_loader))
                    v_mic, v_ref, v_clean, v_vad_gt = v_mic.to(device), v_ref.to(device), v_clean.to(device), v_vad_gt.to(device)
                    v_est = model(v_mic, v_ref)
                    va, vv = compute_loss(v_est, v_clean, v_vad_gt, vad_model, CONFIG)
                    wandb.log({"val/aec_loss": va.item(), "val/vad_loss": vv.item()}, step=global_step)
                model.train()

        torch.save(model.state_dict(), f"aec_vad_final.pth")
    wandb.finish()

if __name__ == "__main__":
    train()
