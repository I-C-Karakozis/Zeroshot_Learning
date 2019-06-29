import os
import argparse

import numpy as np

from tools.log_utils import plot_confusion_matrix

"""
Evalution script:

example execution:
    python eval_awa.py --gt annotations/test_images.txt --pred predictions/l2_outputs.txt

see example_submission.txt for correct submission format

"""

def read_animal_file(fname):
    image_label_dict = {}
    with open(fname) as f:
        for line in f:
            image, label = line.split()
            image_label_dict[image] = label

    return image_label_dict

parser = argparse.ArgumentParser()
parser.add_argument('--gt', help="ground truth labels")
parser.add_argument('--pred', help="file of predictions")
args = parser.parse_args()

# get groundtruth and predicted labels
gt_dict = read_animal_file(args.gt)
pred_dict = read_animal_file(args.pred)

# collect animal classes
names = np.unique([gt_dict[img] for img in gt_dict])
name2id = {}
for i, name in enumerate(names):
    name2id[name] = i
print(name2id)

per_class_accuracy = {"all": []}
gt_labels = []
pred_labels = []

# measure accuracy
for image in gt_dict:
    if image not in pred_dict:
        print("Error: {} not in prediction file".format(image))
        raise Exception()

    gt_label = gt_dict[image]
    gt_labels.append(gt_label)
    pred_label = pred_dict[image]
    pred_labels.append(pred_label)

    if gt_label == pred_label:
        per_class_accuracy["all"].append(1)
    else:
        per_class_accuracy["all"].append(0)
print("Final Accuracy: {:.2f}".format(100 * np.mean(per_class_accuracy["all"])))

# create confusion matrix
gt_id_labels   = np.array([name2id[label] for label in gt_labels])
pred_id_labels = np.array([name2id[label] for label in pred_labels])
accuracy = np.mean(np.equal(gt_id_labels, pred_id_labels))
print("Final Accuracy Verified: {:.2f}".format(100 * accuracy))

plot_confusion_matrix(gt_id_labels, pred_id_labels, names, normalize=False)
plot_confusion_matrix(gt_id_labels, pred_id_labels, names, normalize=True)
