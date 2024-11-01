import os


full_dir = "data/full/"
train_dir = "data/train/"
val_dir = "data/val/"

name = "war_and_peace.txt"

with open(os.path.join(full_dir, name), "r") as f:
    text = f.read()

length = len(text)
split = int(0.8 * length)
train_split = text[:split]
test_split = text[split:]

with open(os.path.join(train_dir, name), "w") as f:
    f.write(train_split)

with open(os.path.join(val_dir, name), "w") as f:
    f.write(test_split)