import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import K_1lead_cls_Dataset
from utils.net1d import Net1D

def build_model(num_lead, base_ckpt, n_classes, device):
    model = Net1D(
        in_channels=num_lead,
        base_filters=64,
        ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes
    )
    checkpoint = torch.load(base_ckpt, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dense.")}
    model.load_state_dict(state_dict, strict=False)
    model.dense = torch.nn.Linear(model.dense.in_features, n_classes).to(device)
    model.to(device)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--df-label-path", required=True)
    parser.add_argument("--ecg-path", required=True)
    parser.add_argument("--tasks", default="hyper_class")
    parser.add_argument("--ecgfounder-ckpt", required=True)
    parser.add_argument("--finetuned-ckpt", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    n_classes = len(tasks)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = build_model(1, args.ecgfounder_ckpt, n_classes, device)
    checkpoint = torch.load(args.finetuned_ckpt, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    df_label = pd.read_csv(args.df_label_path)
    dataset = K_1lead_cls_Dataset(ecg_path=args.ecg_path, labels_df=df_label)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    all_pred = []
    all_gt = []
    all_values = []

    with torch.no_grad():
        for batch in loader:
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

if __name__ == "__main__":
    main()
