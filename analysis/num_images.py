"""
Visualize the number of images for different categories.
See Fig. 2 in the paper.
"""
import os
import json
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import sys

sys.path.append(".")
from common import FACE_ANNOTATIONS_PATH

num_images_with_faces = defaultdict(int)
num_total_images = defaultdict(int)
num_faces_per_image = {}

for annot in tqdm(json.load(open(FACE_ANNOTATIONS_PATH))):
    url = annot["url"]
    if "train" not in url:
        continue
    wnid = url.split(os.path.sep)[-2]
    num_images_with_faces[wnid] += annot["bboxes"] != []
    num_total_images[wnid] += 1
    num_faces_per_image[url] = len(annot["bboxes"])

# Draw Fig. 2 Left
assert len(num_images_with_faces) == len(num_total_images) == 1000
frac_images_with_faces = sorted(
    [num_images_with_faces[wnid] / num_total_images[wnid] for wnid in num_total_images],
    reverse=True,
)
plt.scatter(range(1000), frac_images_with_faces)
plt.hlines(y=[0.25], xmin=0, xmax=216)
plt.hlines(y=[0.50], xmin=0, xmax=106)
plt.vlines(x=[106], ymin=0, ymax=0.5)
plt.vlines(x=[216], ymin=0, ymax=0.25)
plt.xlim([0, 1000])
plt.ylim([0, 1])
plt.savefig("analysis/num_images_per_category.jpg")
print("Figure saved to analysis/num_images_per_category.jpg")

# Draw Fig. 2 Right
plt.figure()
plt.hist([n for n in num_faces_per_image.values() if n > 0], bins=range(1, 11))
plt.savefig("analysis/num_faces_per_image.jpg")
print("Figure saved to analysis/num_faces_per_image.jpg")
