import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.dataset import Custom1DDataset
from utils.net1d import Net1D

def build_model(in_channels, n_classes, ckpt_path, device):
    """Build 1D model and load the fine-tuned checkpoint."""
    model = Net1D(
        in_channels=in_channels,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes
    )
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint saving formats
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference Script for 1D Signals")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    
    # Generalized arguments matching the open-source train.py
    parser.add_argument("--labels-path", required=True, help="Path to the labels/metadata CSV file")
    parser.add_argument("--data-dir", required=True, help="Directory containing 1D signal data files")
    parser.add_argument("--target-cols", default="label", help="Comma-separated target column names")
    
    # Default to the checkpoint folder as requested
    parser.add_argument("--ckpt-path", default="checkpoint/checkpoint.pth", help="Path to the model checkpoint")
    parser.add_argument("--output-csv", default="predictions.csv", help="Path to save the output prediction CSV")
    
    return parser.parse_args()

def main():
    args = parse_args()
    tasks = [t.strip() for t in args.target_cols.split(",") if t.strip()]
    n_classes = len(tasks)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. Initialize model and load weights
    in_channels = 1  # Standard for single-lead signals
    model = build_model(in_channels, n_classes, args.ckpt_path, device)

    # 2. Prepare data
    df_label = pd.read_csv(args.labels_path)
    dataset = Custom1DDataset(data_dir=args.data_dir, labels_df=df_label)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    all_pred = []
    all_gt = []
    all_values = []

    # 3. Run Inference
    print(f"Starting inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in loader:
            # Check if dataset returns original_values along with x and y
            if len(batch) >= 3:
                input_x, input_y, input_values = batch[:3]
                all_values.append(input_values.cpu().numpy())
            else:
                input_x, input_y = batch[:2]
                
            input_x = input_x.to(device)
            input_y = input_y.to(device)
            
            logits = model(input_x)
            pred = torch.sigmoid(logits)
            
            all_pred.append(pred.cpu().numpy())
            all_gt.append(input_y.cpu().numpy())

    pred_arr = np.concatenate(all_pred)
    gt_arr = np.concatenate(all_gt)

    # 4. Format and save outputs
    output = {}
    for i, task in enumerate(tasks):
        if gt_arr.ndim > 1:
            output[f"true_label_{task}"] = gt_arr[:, i]
        else:
            output[f"true_label_{task}"] = gt_arr
            
        if pred_arr.ndim > 1:
            output[f"pred_prob_{task}"] = pred_arr[:, i]
        else:
            output[f"pred_prob_{task}"] = pred_arr

    if len(all_values) > 0:
        values_arr = np.concatenate(all_values)
        if values_arr.ndim > 1 and values_arr.shape[1] == 1:
            values_arr = values_arr[:, 0]
        output["original_value"] = values_arr

    out_df = pd.DataFrame(output)
    out_df.to_csv(args.output_csv, index=False, float_format="%.6f")
    print(f"Inference completed. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()