# file_test.py
from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata

root = "data/iNat2018"
train_ds = get_inat2018(root, "train")
val_ds = get_inat2018(root, "val")

print("Train len:", len(train_ds))
print("Val len:", len(val_ds))

md = extract_hierarchical_metadata(root)
print("First class metadata:", md[0])
