import os
import numpy as np 
import torch as torch
import torch.utils.data as data

from PIL import Image

CLASS_INDEX = -1

def get_all_classes(classes_txt):
    classes = []
    with open(classes_txt) as f:
        for line in f.readlines():
            classes.append(line.split()[-1].strip())

    return classes

def get_classes(classes_txt):
    classes = []
    with open(classes_txt) as f:
        for line in f.readlines():
            classes.append(line.strip())

    return classes

def get_attributes(attributes_txt):
    id2attribute = {}
    with open(attributes_txt) as f:
        for line in f.readlines():
            id_, attr = line.split()
            id2attribute[int(id_)-1] = attr.strip()

    return id2attribute

def get_class_attributes(all_classes, class_attr_matrix_txt):
    class2attributes = {}
    class_id = 0

    with open(class_attr_matrix_txt) as f:
        for line in f.readlines():
            attr_labels = [int(l) for l in line.split()]
            class2attributes[all_classes[class_id]] = attr_labels
            class_id += 1

    return class2attributes

class AWA(data.Dataset):
    def __init__(self, datadir='data/AwA_128x128', transform=None, split=0):

        # annotations txt files
        self.all_classes_txt = 'annotations/classes.txt'
        self.train_classes_txt = 'annotations/trainclasses.txt'
        self.test_classes_txt = 'annotations/testclasses.txt'
        self.attributes_txt = 'annotations/predicates.txt'
        self.class_attr_matrix_txt = 'annotations/predicate-matrix-binary.txt'

        # parse txt files
        self.all_classes   = get_all_classes(self.all_classes_txt)
        self.train_classes = get_classes(self.train_classes_txt)
        self.test_classes  = get_classes(self.test_classes_txt)
        self.id2attribute  = get_attributes(self.attributes_txt)
        self.class2attributes = get_class_attributes(self.all_classes, self.class_attr_matrix_txt)

        self.n_attributes = len(self.id2attribute)

        self.dataset_size = 0
        self.lbls = []
        self.img_paths = []

        self.datadir = datadir
        self.split = split
        self.transform = transform

        if self.split == 0:
            self.classes = self.train_classes
            print("LOADING TRAIN SET...")
        elif self.split == 1:
            self.classes = self.test_classes
            print("LOADING TEST SET...")
        else:
            exit("Bad split id input.")

        self.get_annotations()
        print("{} Images.".format(self.__len__()))

    def get_annotations(self):
        for i, class_ in enumerate(self.classes):
            attr_labels = self.class2attributes[class_]

            # get image paths
            class_dir = os.path.join(self.datadir, class_)
            class_img_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            self.img_paths = self.img_paths + class_img_paths

            # get labels
            class_lbls = attr_labels + [i]
            self.lbls = self.lbls + ([class_lbls] * len(class_img_paths))            
        
        self.dataset_size = len(self.lbls)
        assert(len(self.img_paths) == len(self.lbls))
        assert(len(self.lbls[0]) == self.n_attributes + 1) 

    def get_attribute_vectors(self):
        attr_vectors = []
        for class_ in self.classes:
            attr_labels = self.class2attributes[class_]
            attr_vectors.append(attr_labels)
            
        return attr_vectors

    def __getitem__(self, idx):
        # preprocess image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return torch.FloatTensor(img), torch.FloatTensor(np.array(self.lbls[idx]).astype('uint8'))

    def __len__(self):
        return self.dataset_size

if __name__ == "__main__":
    dataset_train = AWA(split=0)
    dataset_test  = AWA(split=1)
