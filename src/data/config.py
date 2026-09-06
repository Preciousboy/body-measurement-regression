from pathlib import Path

Root_dir = Path(__file__).resolve().parents[2]

# Path to the metadata files

Train_csv = Root_dir/f"metadata/bodym_train.csv"
Val_csv = Root_dir/f"metadata/bodym_val.csv"
Test_csv = Root_dir/f"metadata/bodym_test.csv"

# Path to the actual image files

Train_mask_dir = Root_dir/f"data/train/mask"
Train_mask_left_dir = Root_dir/f"data/train/mask_left"

Val_mask_dir = Root_dir/f"data/val/mask"
Val_mask_left_dir = Root_dir/f"data/val/mask_left"

Test_mask_dir = Root_dir/f"data/test/mask"
Test_mask_left_dir = Root_dir/f"data/test/mask_left"

# Path to  checkpoint
Checkpoint_dir = Root_dir/f"checkpoints"

# from PIL import Image
# image_mask = Image.open(Train_mask_dir/f"fffff15c2d2d77492da0d0d13e013774.png")
# image_mask_left = Image.open(Train_mask_left_dir/f"ffe009aa411dc2e13d657f9717243928.png")

# print(image_mask)
# print("size:", image_mask.size)
# print("mode", image_mask.mode)
# image_mask.show()

# print(image_mask_left)
# print("size:", image_mask_left.size)
# print("mode", image_mask_left.mode)
# image_mask_left.show()

