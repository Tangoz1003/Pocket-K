import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Custom1DDataset(Dataset):
    """
    A unified PyTorch Dataset for 1D signal data (e.g., single-lead ECG).
    Handles loading, lead extraction, dynamic padding/cropping, and normalization.
    """
    def __init__(self, data_dir, labels_df, target_cols=["label"], transform=None):
        """
        Args:
            data_dir (str): Directory path containing the .npy data files.
            labels_df (DataFrame): DataFrame containing annotations.
            target_cols (list): List of column names to be used as prediction targets.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.target_cols = target_cols
        self.transform = transform
        
        # Standard 12-lead names
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # Leads to extract for the model (Lead I by default)
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self, signal):
        """Normalize signal to zero mean and unit variance."""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8) 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        record = self.labels_df.iloc[idx]
        file_path = record.get("file_path", "")
        
        # Resolve absolute path
        if self.data_dir and not os.path.isabs(file_path):
            file_path = os.path.join(self.data_dir, file_path)
            
        # Extract ground truth values and labels
        values = record.get("original_value", 0.0)
        values = np.array(values, dtype=np.float32)
        
        # Dynamically load target labels based on provided column names
        if len(self.target_cols) == 1 and self.target_cols[0] in record:
            labels = record[self.target_cols[0]]
        else:
            labels = record.get("label", 0.0) # Fallback to standard 'label' column
        labels = np.array(labels, dtype=np.float32)

        # Load 1D signal data
        try:
            data = np.load(file_path) # Expected shape: (time_steps, channels) e.g., (12000, 12)
        except FileNotFoundError:
            # Fallback to zeros if file is missing to prevent batch failure
            data = np.zeros((12000, 12), dtype=np.float32)
            
        data = np.nan_to_num(data, nan=0.0)
        
        # Transpose: (12000, 12) -> (12, 12000)
        data = np.transpose(data, (1, 0)) 
        
        # Extract specific leads: (12, 12000) -> (1, 12000)
        data = data[self.lead_indices, :] 

        # Center crop or pad to exactly 5000 time steps
        target_len = 5000
        current_len = data.shape[1] 
        
        if current_len > target_len:
            # Center crop
            start = (current_len - target_len) // 2
            end = start + target_len
            data = data[:, start:end]
            
        elif current_len < target_len:
            # Zero-pad at the end
            pad_len = target_len - current_len
            data = np.pad(data, ((0, 0), (0, pad_len)), mode='constant')

        # Z-score normalization
        signal_data = self.z_score_normalization(data)
        signal_tensor = torch.FloatTensor(signal_data)

        # Format output tensors
        values_tensor = torch.tensor(values, dtype=torch.float)
        if values_tensor.dim() == 0:
            values_tensor = values_tensor.unsqueeze(0) 

        labels_tensor = torch.tensor(labels, dtype=torch.float)
        if labels_tensor.dim() == 0:  
            labels_tensor = labels_tensor.unsqueeze(0)
            
        return signal_tensor, labels_tensor, values_tensor