"""
Table B in the paper.
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

# (model, overlayred) -> [seed1, seed2, seed3]
top1_accuracies: Dict[Tuple[str, bool], List[float]] = defaultdict(lambda: [0, 0, 0])
top5_accuracies: Dict[Tuple[str, bool], List[float]] = defaultdict(lambda: [0, 0, 0])

for f in tqdm(glob(os.path.join(EVAL_PICKLES_PATH, "eval_*_val_results.pickle"))):
    fields = f.split("_")
    seed = int(fields[-3][4:])
    if "blur" in f:
        continue
    elif "overlaytrain" in f and "overlayval" in f:
        overlayred = True
        model = "_".join(fields[2:-5])
    elif "overlaytrain" not in f and "overlayval" not in f:
        overlayred = False
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

    top1_accuracies[(model, overlayred)][seed - 1] = top1_acc
    top5_accuracies[(model, overlayred)][seed - 1] = top5_acc

# Print the table
table = []
top1_accuracies_original = []
top5_accuracies_original = []
top1_accuracies_overlayred = []
top5_accuracies_overlayred = []
for model in all_models:
    top1_original = 100 * np.mean(top1_accuracies[(model, False)])
    top1_original_sem = 100 * scipy.stats.sem(top1_accuracies[(model, False)], ddof=2)
    top1_overlayred = 100 * np.mean(top1_accuracies[(model, True)])
    top1_overlayred_sem = 100 * scipy.stats.sem(top1_accuracies[(model, True)], ddof=2)
    top5_original = 100 * np.mean(top5_accuracies[(model, False)])
    top5_original_sem = 100 * scipy.stats.sem(top5_accuracies[(model, False)], ddof=2)
    top5_overlayred = 100 * np.mean(top5_accuracies[(model, True)])
    top5_overlayred_sem = 100 * scipy.stats.sem(top5_accuracies[(model, True)], ddof=2)
    table.append(
        [
            model,
            "%.03f +- %.03f" % (top1_original, top1_original_sem),
            "%.03f +- %.03f" % (top1_overlayred, top1_overlayred_sem),
            "%.03f" % (top1_original - top1_overlayred),
            "%.03f +- %.03f" % (top5_original, top5_original_sem),
            "%.03f +- %.03f" % (top5_overlayred, top5_overlayred_sem),
            "%.03f" % (top5_original - top5_overlayred),
        ]
    )
    top1_accuracies_original.append(top1_original)
    top5_accuracies_original.append(top5_original)
    top1_accuracies_overlayred.append(top1_overlayred)
    top5_accuracies_overlayred.append(top5_overlayred)

avg_top1_original = np.mean(top1_accuracies_original)
avg_top1_overlayred = np.mean(top1_accuracies_overlayred)
avg_top5_original = np.mean(top5_accuracies_original)
avg_top5_overlayred = np.mean(top5_accuracies_overlayred)
table.append(
    [
        "average",
        "%.03f" % avg_top1_original,
        "%.03f" % avg_top1_overlayred,
        "%.03f" % (avg_top1_original - avg_top1_overlayred),
        "%.03f" % avg_top5_original,
        "%.03f" % avg_top5_overlayred,
        "%.03f" % (avg_top5_original - avg_top5_overlayred),
    ]
)

print(
    tabulate(
        table,
        headers=[
            "model",
            "top1 original",
            "top1 overlayred",
            "top1 diff",
            "top5 original",
            "top5 overlayred",
            "top5 diff",
        ],
    )
)
