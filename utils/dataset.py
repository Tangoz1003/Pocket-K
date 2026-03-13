import os
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.signal import resample


class K_1lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        record = self.labels_df.iloc[idx]
        file_path = record["file_path"]
        if self.ecg_path and not os.path.isabs(file_path):
            file_path = os.path.join(self.ecg_path, file_path)
        values = record["original_value"]
        values = np.array(values, dtype=np.float32)
        labels = record["label"]
        labels = np.array(labels, dtype=np.float32)

        # 2. 加载数据
        try:
            data = np.load(file_path) # 形状 (12000, 12)
        except FileNotFoundError:
            # 容错：直接返回一个符合目标形状的全0数组
            data = np.zeros((12000, 12), dtype=np.float32)
            
        data = np.nan_to_num(data, nan=0)
        
        # 3. 维度转置 (12000, 12) -> (12, 12000)
        # 必须先转置，把 '时间' 放到第1维，方便切片
        data = np.transpose(data, (1, 0)) 
        
        # 4. 提取导联 (12, 12000) -> (1, 12000)
        data = data[self.lead_indices, :] 

        # ==========================================
        # 【核心修改区】 5000点 截取/补零 逻辑
        # ==========================================
        target_len = 5000
        current_len = data.shape[1] # 现在是 12000
        
        if current_len > target_len:
            # A. 超过5000：取中间
            start = (current_len - target_len) // 2
            end = start + target_len
            data = data[:, start:end]
            
        elif current_len < target_len:
            # B. 不足5000：末尾补0 (也可以改为两边补，这里用末尾补0最稳妥)
            pad_len = target_len - current_len
            # ((0,0), (0, pad_len)) 表示：第0维(导联)不补，第1维(时间)右侧补 pad_len 个 0
            data = np.pad(data, ((0,0), (0, pad_len)), mode='constant')
            
        # C. 刚好5000：不做处理，直接用
        # ==========================================

        # 5. 归一化 (此时 data 形状固定为 (1, 5000))
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # 6. 标签维度处理 (保持不变)
        values = torch.tensor(values, dtype=torch.float)
        if values.dim() == 0:
            values = values.unsqueeze(0) 

        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
            
        return signal, labels, values
    
