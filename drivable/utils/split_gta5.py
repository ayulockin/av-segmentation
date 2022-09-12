import glob
import os
from sklearn.model_selection import train_test_split
import random
import numpy as np

img_paths = glob.glob("/home/manan_goel/GTA5/images/*.png")
img_paths = [path.split("/")[-1] for path in img_paths]
train_paths, val_paths = train_test_split(img_paths, train_size=0.8, shuffle=True)

np.savetxt(
    "./split_gta5/train_split.txt",
    train_paths,
    fmt="%s",
    delimiter=","
)

np.savetxt(
    "./split_gta5/val_split.txt",
    val_paths,
    fmt="%s",
    delimiter=","
)