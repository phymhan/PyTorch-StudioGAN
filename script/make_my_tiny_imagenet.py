import os
import shutil
from tqdm import tqdm

with open('imagenet100.txt', 'r') as f:
    cats = f.readlines()

src = '/dresden/users/lh599/Data/ILSVRC2012/train'
dst = '/dresden/users/lh599/Data/ImageNet100/train'

for c in tqdm(cats):
    c = c.rstrip()
    os.system(f"cp -r {os.path.join(src, c)} {os.path.join(dst, c)}")
