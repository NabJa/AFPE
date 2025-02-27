""""This script generates the main results (Table 1) of the paper 'Principled Positional Encodings for Medical Imaging'."""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
)
from torchmetrics.wrappers import BootStrapper


def find_results_folder():
    # Find the results folder
    results_folder = Path("./results")
    if results_folder.exists():
        return results_folder
    
    results_folder = Path("../results")
    if results_folder.exists():
        return results_folder
    
    raise FileNotFoundError(f"""Could not find the results folder.
                            Make sure it is in the directory you call this script from.
                            Your current working directiony is {Path.cwd()}.""")


def generate_chestx_results(result_dir: Path, chest_results_name = "chestx_predictions.csv"):
    df = pd.read_csv(result_dir / chest_results_name, index_col=0)

    metric_collection = MetricCollection({
        "AUPRC":   BootStrapper(MultilabelAveragePrecision(num_labels=20), num_bootstraps=100), # Suited metric!
        "F1":    BootStrapper(MultilabelF1Score(num_labels=20), num_bootstraps=3), # Not recommended, because of single threshold!
        "AUROC": BootStrapper(MultilabelAUROC(num_labels=20), num_bootstraps=3) # Not recommended, because of class imbalance!
    })

    results = defaultdict(list)

    for model, group in df.groupby("model"):
        # Convert to tensors of shape (num_samples, num_labels)
        y_true = torch.tensor(group.pivot(index="sample", columns="label", values="y").values, dtype=torch.int)
        y_pred = torch.tensor(group.pivot(index="sample", columns="label", values="pred").values, dtype=torch.float)
        
        # Re-seeding before calling metric_collection(y_pred, y_true) 
        # ensures that any random sampling done within BootStrapper is deterministic
        random.seed(4242)
        np.random.seed(4242)
        torch.manual_seed(4242)

        metrics = metric_collection(y_pred, y_true)
        
        for metric in metric_collection.keys():
            results["metric"].append(metric)
            results["mean"].append(metrics[f"{metric}_mean"].item())
            results["std"].append(metrics[f"{metric}_std"].item())
            results["model"].append(model)

    results = pd.DataFrame(results)

    best_models = results[results.metric == "AUPRC"].sort_values("mean", ascending=False).reset_index(drop=True)
    
    return best_models

def generate_echonet_results(result_dir: Path, echonet_results_name = "echonet_predictions.csv"):
    df = pd.read_csv(result_dir / echonet_results_name)

    metric_collection = MetricCollection({
        "MAE": BootStrapper(MeanAbsoluteError(), num_bootstraps=1000),
        "RMSE": BootStrapper(MeanSquaredError(squared=False), num_bootstraps=2000),
        "R2": BootStrapper(R2Score(), num_bootstraps=2000)
    })
    
    results = defaultdict(list)
    for model, group in df.groupby("model"):
        y_true = torch.tensor(group.y.values, dtype=torch.float)
        y_pred = torch.tensor(group.pred.values, dtype=torch.float)

        # Re-seeding before calling metric_collection(y_pred, y_true) 
        # ensures that any random sampling done within BootStrapper is deterministic
        random.seed(4242)
        np.random.seed(4242)
        torch.manual_seed(4242)
        
        metrics = metric_collection(y_pred, y_true)

        for metric in metric_collection.keys():
            results["metric"].append(metric)
            results["mean"].append(metrics[f"{metric}_mean"].item())
            results["std"].append(metrics[f"{metric}_std"].item())
            results["model"].append(model)
    results = pd.DataFrame(results)

    best_models = results[results.metric == "R2"].sort_values("mean", ascending=False)
    
    return best_models


if __name__ == "__main__":

    result_dir = find_results_folder()

    print("Generating results for echonet...")
    best_echonet_models = generate_echonet_results(result_dir)
    print("Best Echonet models:")
    print(best_echonet_models.round(3))

    print("\nGenerating results for chestx...")
    best_chestx_models = generate_chestx_results(result_dir)
    print("Best ChestX models:")
    print(best_chestx_models.round(3))
