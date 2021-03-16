"""
Table 3 in the paper.
"""
import os
import pickle
from glob import glob
from tqdm import tqdm
import numpy as np
import scipy.stats
from tabulate import tabulate
from collections import defaultdict
from typing import List, Dict, Tuple
import sys

sys.path.append(".")
from common import EVAL_PICKLES_PATH


all_models = [
    "alexnet",
    "squeezenet1_0",
    "shufflenet_v2_x1_0",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "mobilenet_v2",
    "densenet121",
    "densenet201",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

# (model, blurred) -> [seed1, seed2, seed3]
top1_accuracies: Dict[Tuple[str, bool], List[float]] = defaultdict(lambda: [0, 0, 0])
top5_accuracies: Dict[Tuple[str, bool], List[float]] = defaultdict(lambda: [0, 0, 0])

for f in tqdm(glob(os.path.join(EVAL_PICKLES_PATH, "eval_*_val_results.pickle"))):
    fields = f.split("_")
    seed = int(fields[-3][4:])
    if "overlay" in f:
        continue
    elif "blurtrain" in f and "blurval" in f:
        blurred = True
        model = "_".join(fields[2:-5])
    elif "blurtrain" not in f and "blurval" not in f:
        blurred = False
        model = "_".join(fields[2:-3])
    else:
        continue
    assert model in all_models

    val_results = pickle.load(open(f, "rb"))
    top1_acc = 0.0
    top5_acc = 0.0
    for example in val_results["examples"]:
        top1_acc += example["label"] == example["top5_predictions"][0]
        top5_acc += example["label"] in example["top5_predictions"]
    num_examples = len(val_results["examples"])
    top1_acc /= num_examples
    top5_acc /= num_examples

    top1_accuracies[(model, blurred)][seed - 1] = top1_acc
    top5_accuracies[(model, blurred)][seed - 1] = top5_acc

# Print the table
table = []
top1_accuracies_original = []
top5_accuracies_original = []
top1_accuracies_blurred = []
top5_accuracies_blurred = []
for model in all_models:
    top1_original = 100 * np.mean(top1_accuracies[(model, False)])
    top1_original_sem = 100 * scipy.stats.sem(top1_accuracies[(model, False)], ddof=2)
    top1_blurred = 100 * np.mean(top1_accuracies[(model, True)])
    top1_blurred_sem = 100 * scipy.stats.sem(top1_accuracies[(model, True)], ddof=2)
    top5_original = 100 * np.mean(top5_accuracies[(model, False)])
    top5_original_sem = 100 * scipy.stats.sem(top5_accuracies[(model, False)], ddof=2)
    top5_blurred = 100 * np.mean(top5_accuracies[(model, True)])
    top5_blurred_sem = 100 * scipy.stats.sem(top5_accuracies[(model, True)], ddof=2)
    table.append(
        [
            model,
            "%.03f +- %.03f" % (top1_original, top1_original_sem),
            "%.03f +- %.03f" % (top1_blurred, top1_blurred_sem),
            "%.03f" % (top1_original - top1_blurred),
            "%.03f +- %.03f" % (top5_original, top5_original_sem),
            "%.03f +- %.03f" % (top5_blurred, top5_blurred_sem),
            "%.03f" % (top5_original - top5_blurred),
        ]
    )
    top1_accuracies_original.append(top1_original)
    top5_accuracies_original.append(top5_original)
    top1_accuracies_blurred.append(top1_blurred)
    top5_accuracies_blurred.append(top5_blurred)

avg_top1_original = np.mean(top1_accuracies_original)
avg_top1_blurred = np.mean(top1_accuracies_blurred)
avg_top5_original = np.mean(top5_accuracies_original)
avg_top5_blurred = np.mean(top5_accuracies_blurred)
table.append(
    [
        "average",
        "%.03f" % avg_top1_original,
        "%.03f" % avg_top1_blurred,
        "%.03f" % (avg_top1_original - avg_top1_blurred),
        "%.03f" % avg_top5_original,
        "%.03f" % avg_top5_blurred,
        "%.03f" % (avg_top5_original - avg_top5_blurred),
    ]
)

print(
    tabulate(
        table,
        headers=[
            "model",
            "top1 original",
            "top1 blurred",
            "top1 diff",
            "top5 original",
            "top5 blurred",
            "top5 diff",
        ],
    )
)
