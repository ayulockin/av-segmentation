import os
import numpy as np

# We will create a 30K random split for training + 10K of the rest 40K for validation
TRAIN_DATA_PATH = f"artifacts/bdd100k-dataset:v0/images/100k/train"
TEST_DATA_PATH = f"artifacts/bdd100k-dataset:v0/images/100k/val"

# We will create a 30K random split for training + 10K of the rest 40K for validation
TRAIN_MASK_PATH = f"artifacts/train_masks:v0"
VALID_MASK_PATH = f"artifacts/val_masks:v0"

train_img_files = os.listdir(TRAIN_DATA_PATH)
test_img_files = os.listdir(TEST_DATA_PATH)

# TRAIN SPLIT
# Random sample of images
print("Number of train files before split: ", len(set(train_img_files)))
train_30k_files = np.random.choice(train_img_files, size=30000, replace=False)

# VAL SPLIT
train_40k_files = list(set(train_img_files) - set(train_30k_files))
print("Number of val files before split: ", len(set(train_40k_files)))
val_10k_files = np.random.choice(train_img_files, size=30000, replace=False)

# TEST SPLIT
print("Number of test files before split: ", len(set(test_img_files)))
test_10k_files = test_img_files


os.makedirs("splits")
# Save the split as txt file
np.savetxt(f"splits/train_split.txt", train_30k_files, fmt="%s", delimiter=",")

np.savetxt(f"splits/val_split.txt", val_10k_files, fmt="%s", delimiter=",")

np.savetxt(f"splits/test_split.txt", test_10k_files, fmt="%s", delimiter=",")
