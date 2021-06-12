import os
import shutil
root = 'data/ILSVRC2012/valid'
for i in os.listdir(root):
    os.rename(os.path.join(root, i), os.path.join(root, i.replace('.JPEG', '.jpeg')))

