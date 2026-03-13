import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.dataset import K_1lead_cls_Dataset
from utils.net1d import Net1D
from utils.util import save_checkpoint, my_eval_with_dynamic_thresh, bootstrap_ci

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_model(num_lead, base_ckpt, n_classes, device, linear_prob):
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
    model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)
    if linear_prob:
        for name, param in model.named_parameters():
            if "dense" not in name:
                param.requires_grad = False
    model.to(device)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stop-lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--df-label-path", required=True)
    parser.add_argument("--ecg-path", required=True)
    parser.add_argument("--tasks", default="hyper_class")
    parser.add_argument("--saved-dir", required=True)
    parser.add_argument("--ecgfounder-ckpt", required=True)
    parser.add_argument("--linear-prob", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.saved_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    n_classes = len(tasks)

    model = build_model(1, args.ecgfounder_ckpt, n_classes, device, args.linear_prob)
    df_label = pd.read_csv(args.df_label_path)
    train_df = df_label.sample(frac=0.8, random_state=args.seed)
    val_df = df_label.drop(train_df.index)

    train_dataset = K_1lead_cls_Dataset(ecg_path=args.ecg_path, labels_df=train_df)
    val_dataset = K_1lead_cls_Dataset(ecg_path=args.ecg_path, labels_df=val_df)

    target_col = tasks[0]
    targets = train_df[target_col].values
    class_counts = np.bincount(targets.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets.astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode="max", verbose=True)

    best_val_auroc = 0.
    step = 0
    all_val = []

    total_steps_per_epoch = len(trainloader)
    eval_steps = total_steps_per_epoch

    for epoch in range(args.epochs):
        model.train()
        for batch in trainloader:
            input_x, input_y, input_original_values = batch[:3]
            input_x = input_x.to(device)
            input_y = input_y.to(device)
            input_original_values = input_original_values.to(device)
            outputs = model(input_x)
            loss = criterion(outputs, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % eval_steps == 0:
                model.eval()
                all_gt = []
                all_pred_prob = []
                all_original_values = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(valloader):
                        input_x, input_y, input_original_values = batch[:3]
                        input_x = input_x.to(device)
                        input_y = input_y.to(device)
                        input_original_values = input_original_values.to(device)
                        logits = model(input_x)
                        pred = torch.sigmoid(logits)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                        all_original_values.append(input_original_values.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                all_original_values = np.concatenate(all_original_values)
                all_original_values = np.array(all_original_values)
                res_val, res_val_auroc, res_val_sens, res_val_spec, res_val_f1, res_val_auprc, thre = my_eval_with_dynamic_thresh(all_gt, all_pred_prob)
                auroc_cis = []
                for i in range(n_classes):
                    gt_i = all_gt[:, i] if all_gt.ndim > 1 else all_gt
                    pred_i = all_pred_prob[:, i] if all_pred_prob.ndim > 1 else all_pred_prob
                    gt_i = np.nan_to_num(gt_i, nan=0)
                    pred_i = np.nan_to_num(pred_i, nan=0)
                    ci_low, ci_high = bootstrap_ci(gt_i, pred_i, metric="roc_auc")
                    auroc_cis.append((ci_low, ci_high))
                val_auroc = res_val
                is_best = bool(val_auroc > best_val_auroc)
                if is_best:
                    best_val_auroc = val_auroc
                    save_checkpoint({
                        "epoch": epoch,
                        "step": step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_auroc": val_auroc,
                    }, args.saved_dir)
                    if len(tasks) == 1:
                        predictions_df = pd.DataFrame({
                            "true_label": all_gt.flatten(),
                            "pred_prob": all_pred_prob.flatten(),
                            "original_value": all_original_values.flatten()
                        })
                        predictions_df.to_csv(os.path.join(args.saved_dir, "best_predictions.csv"), index=False, float_format="%.5f")
                    else:
                        for i, task in enumerate(tasks):
                            task_df = pd.DataFrame({
                                "true_label": all_gt[:, i],
                                "pred_prob": all_pred_prob[:, i],
                                "original_value": all_original_values[:, i]
                            })
                            task_df.to_csv(os.path.join(args.saved_dir, f"best_predictions_{task}.csv"), index=False, float_format="%.5f")
                for i, task in enumerate(tasks):
                    pos_count = val_df[task].sum()
                    neg_count = len(val_df) - pos_count
                    all_val.append([
                        task, res_val_auroc[i],
                        f"{auroc_cis[i][0]:.4f}-{auroc_cis[i][1]:.4f}",
                        res_val_sens[i], res_val_spec[i],
                        res_val_f1[i], res_val_auprc[i], thre[i], pos_count, neg_count
                    ])
                columns = ["Field_ID", "AUROC", "AUROC_95CI", "sensitivity", "specificity", "f1", "auprc", "thre", "pos_num","neg_num"]
                df = pd.DataFrame(all_val, columns=columns)
                df.to_csv(os.path.join(args.saved_dir, "val.csv"), index=False, float_format="%.5f")
                scheduler.step(val_auroc)
                current_lr = optimizer.param_groups[0]["lr"]
                if current_lr < args.early_stop_lr:
                    return
                model.train()

if __name__ == "__main__":
    main()
