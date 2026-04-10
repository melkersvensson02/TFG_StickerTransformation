import os
import json
import random
import shutil

# -----------------------
# CONFIG
# -----------------------

TEST_A = "test_A"
TEST_B = "test_B"
TRAIN_A = "train_A"
TRAIN_B = "train_B"

TEST_JSON = "test_prompts.json"
TRAIN_JSON = "train_prompts.json"

NUM_VALIDATION = 50  # keep 50 in test

# -----------------------
# STEP 1 — Load test filenames
# -----------------------

test_filenames = sorted(os.listdir(TEST_A))

assert len(test_filenames) == 125, "Expected 125 test images"

# -----------------------
# STEP 2 — Select validation subset
# -----------------------

random.seed(42)  # reproducible
validation_files = set(random.sample(test_filenames, NUM_VALIDATION))

files_to_remove_from_test = set(test_filenames) - validation_files

# -----------------------
# STEP 3 — Remove 75 from test folders
# -----------------------

for filename in files_to_remove_from_test:
    os.remove(os.path.join(TEST_A, filename))
    os.remove(os.path.join(TEST_B, filename))

print(f"Removed {len(files_to_remove_from_test)} images from test.")

# -----------------------
# STEP 4 — Remove validation images from training folders
# -----------------------

for filename in validation_files:
    path_A = os.path.join(TRAIN_A, filename)
    path_B = os.path.join(TRAIN_B, filename)

    if os.path.exists(path_A):
        os.remove(path_A)
    if os.path.exists(path_B):
        os.remove(path_B)

print(f"Removed {len(validation_files)} validation images from training.")

# -----------------------
# STEP 5 — Update test JSON
# -----------------------

with open(TEST_JSON, "r") as f:
    # By default this creates a dict object, json.load does the mapping to dict for us
    # So now to access a value we simply do in O(1) test_prompts["img_001.png"]
    test_prompts = json.load(f)

# This is hard, it works like this k is the key in the new dict 
# and v is a value in the old dict, we only keep the key value pairs where the key is in the validation_files set
new_test_prompts = {
    k: v for k, v in test_prompts.items()
    if k in validation_files
}

with open(TEST_JSON, "w") as f:
    json.dump(new_test_prompts, f, indent=4)

print("Updated test_prompts.json")

# -----------------------
# STEP 6 — Update train JSON
# -----------------------

with open(TRAIN_JSON, "r") as f:
    train_prompts = json.load(f)

new_train_prompts = {
    k: v for k, v in train_prompts.items()
    if k not in validation_files
}

with open(TRAIN_JSON, "w") as f:
    json.dump(new_train_prompts, f, indent=4)

print("Updated train_prompts.json")

print("Done.")