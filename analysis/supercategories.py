"""
Show the number of faces in supercategories.
See Table 2 in the paper.
"""
import os
import json
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from tabulate import tabulate
import sys

sys.path.append(".")
from common import FACE_ANNOTATIONS_PATH

from typing import List, Set

supercategories = [
    wn.synset(s)
    for s in (
        "clothing.n.01",
        "wheeled_vehicle.n.01",
        "musical_instrument.n.01",
        "bird.n.01",
        "insect.n.01",
    )
]

num_supercategories = len(supercategories)
categories: List[Set[str]] = [set() for _ in range(num_supercategories)]
num_images = [0 for _ in range(num_supercategories)]
num_images_with_faces = [0 for _ in range(num_supercategories)]

for annot in tqdm(json.load(open(FACE_ANNOTATIONS_PATH))):
    if "train" not in annot["url"]:
        continue
    wnid = annot["url"].split(os.path.sep)[-2]
    synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    # Get all ancestors in the WordNet hierarchy
    all_hypernyms = set(synset.closure(lambda s: s.hypernyms()))
    all_hypernyms.add(synset)
    # Check if the ancestors including the supercategories we are interested in
    for i, c in enumerate(supercategories):
        if c in all_hypernyms:
            categories[i].add(wnid)
            num_images[i] += 1
            num_images_with_faces[i] += annot["bboxes"] != []

# Print the table
table = []
for i, c in enumerate(supercategories):
    table.append(
        [
            c.name(),
            len(categories[i]),
            num_images[i],
            100 * num_images_with_faces[i] / num_images[i],
        ]
    )

print(
    tabulate(
        table, headers=["Supercategory", "#Categories", "#Images", "With faces (%)"]
    )
)
