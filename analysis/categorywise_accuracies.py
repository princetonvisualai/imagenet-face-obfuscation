"""
Table 4 in the paper.
"""
import os
import pickle
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import scipy.stats
from tabulate import tabulate
from collections import defaultdict
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
import sys

sys.path.append(".")
from common import EVAL_PICKLES_PATH, SORTED_WNIDS_PATH


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

sorted_wnids = json.load(open(SORTED_WNIDS_PATH))

# WordNet ID -> [[0/1 accuracies for different models and images] for different seeds]
top1_original = defaultdict(lambda: [[], [], []])
top1_blurred = defaultdict(lambda: [[], [], []])
top5_original = defaultdict(lambda: [[], [], []])
top5_blurred = defaultdict(lambda: [[], [], []])

# [[1000-d vector of APs for different models] for different seeds]
AP_original = [[], [], []]
AP_blurred = [[], [], []]

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
    gt_labels = []
    for example in val_results["examples"]:
        gt_labels.append(example["label"])
        wnid = sorted_wnids[example["label"]]
        if blurred:
            top1_blurred[wnid][seed - 1].append(
                float(example["label"] == example["top5_predictions"][0])
            )
            top5_blurred[wnid][seed - 1].append(
                float(example["label"] in example["top5_predictions"])
            )
        else:
            top1_original[wnid][seed - 1].append(
                float(example["label"] == example["top5_predictions"][0])
            )
            top5_original[wnid][seed - 1].append(
                float(example["label"] in example["top5_predictions"])
            )

    onehot_labels = F.one_hot(torch.tensor(gt_labels), num_classes=1000).numpy()
    APs = average_precision_score(
        onehot_labels, val_results["output_scores"], average=None
    )
    if blurred:
        AP_blurred[seed - 1].append(APs)
    else:
        AP_original[seed - 1].append(APs)

top1_original = {
    wnid: [np.mean(x) for x in top1_original[wnid]] for wnid in top1_original
}
top1_blurred = {wnid: [np.mean(x) for x in top1_blurred[wnid]] for wnid in top1_blurred}
top5_original = {
    wnid: [np.mean(x) for x in top5_original[wnid]] for wnid in top5_original
}
top5_blurred = {wnid: [np.mean(x) for x in top5_blurred[wnid]] for wnid in top5_blurred}

AP_original = [np.vstack(x).mean(axis=0) for x in AP_original]
AP_blurred = [np.vstack(x).mean(axis=0) for x in AP_blurred]

# Table 4
table_4 = []
categories_of_interest = [
    "eskimo_dog.n.01",
    "siberian_husky.n.01",
    "projectile.n.01",
    "missile.n.01",
    "tub.n.02",
    "bathtub.n.01",
    "american_chameleon.n.01",
    "green_lizard.n.01",
]
for cat in categories_of_interest:
    wnid = "n%08d" % wn.synset(cat).offset()
    top1_original_mean = 100 * np.mean(top1_original[wnid])
    top1_original_sem = 100 * scipy.stats.sem(top1_original[wnid], ddof=2)
    top1_blurred_mean = 100 * np.mean(top1_blurred[wnid])
    top1_blurred_sem = 100 * scipy.stats.sem(top1_blurred[wnid], ddof=2)
    top5_original_mean = 100 * np.mean(top5_original[wnid])
    top5_original_sem = 100 * scipy.stats.sem(top5_original[wnid], ddof=2)
    top5_blurred_mean = 100 * np.mean(top5_blurred[wnid])
    top5_blurred_sem = 100 * scipy.stats.sem(top5_blurred[wnid], ddof=2)
    idx = sorted_wnids.index(wnid)
    AP_original_mean = 100 * np.mean([AP_original[i][idx] for i in range(3)])
    AP_original_sem = 100 * scipy.stats.sem(
        [AP_original[i][idx] for i in range(3)], ddof=2
    )
    AP_blurred_mean = 100 * np.mean([AP_blurred[i][idx] for i in range(3)])
    AP_blurred_sem = 100 * scipy.stats.sem(
        [AP_blurred[i][idx] for i in range(3)], ddof=2
    )
    table_4.append(
        [
            cat,
            "%.03f +- %.03f" % (top1_original_mean, top1_original_sem),
            "%.03f +- %.03f" % (top1_blurred_mean, top1_blurred_sem),
            "%.03f" % (top1_original_mean - top1_blurred_mean),
            "%.03f +- %.03f" % (top5_original_mean, top5_original_sem),
            "%.03f +- %.03f" % (top5_blurred_mean, top5_blurred_sem),
            "%.03f" % (top5_original_mean - top5_blurred_mean),
            "%.03f +- %.03f" % (AP_original_mean, AP_original_sem),
            "%.03f +- %.03f" % (AP_blurred_mean, AP_blurred_sem),
            "%.03f" % (AP_original_mean - AP_blurred_mean),
        ]
    )


print(
    tabulate(
        table_4,
        headers=[
            "Category",
            "top1 original",
            "top1 blurred",
            "top1 diff",
            "top5 original",
            "top5 blurred",
            "top5 diff",
            "AP original",
            "AP blurred",
            "AP diff",
        ],
    )
)
