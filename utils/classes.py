import json
import pathlib
import tempfile

import torchvision
from nltk.corpus import wordnet as wn

PATH_ROOT = pathlib.Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


# CIFAR100 #############################################################################

path_classes = PATH_DATASETS / "CIFAR100" / "classes"

with tempfile.TemporaryDirectory() as tmp_dir:
    classes = torchvision.datasets.CIFAR100(tmp_dir, train=False, download=True).classes

with open(path_classes / "classes.txt", "w") as f:
    f.write("\n".join(classes))


# iNaturalist19 ########################################################################

path_classes = PATH_DATASETS / "iNaturalist19" / "classes"

with open(path_classes / "categories.json") as file:
    classes = [category["name"] for category in json.load(file)]

with open(path_classes / "classes.txt", "w") as f:
    f.write("\n".join(classes))


# tieredImageNet  ######################################################################

path_classes = PATH_DATASETS / "tieredImageNet" / "classes"

with open(path_classes / "dir_to_int.json") as f:
    dir_to_int = json.load(f)
    int_to_dir = {v: k for k, v in dir_to_int.items()}


def dir_to_class(dir_name):
    word_id = int(dir_name[1:])
    return wn.synset_from_pos_and_offset("n", word_id).lemmas()[0].name()


classes = [dir_to_class(int_to_dir[i]) for i in range(len(int_to_dir))]

with open(path_classes / "classes.txt", "w") as f:
    f.write("\n".join(classes))
