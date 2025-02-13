import os
import numpy as np
import random

objects = ["frog"]
# objects = ["airplane", "automobile", "bird", "cat", "deer", "dog", "forg", "horse", "ship", "truck"]
for ob in objects:
    for epoch in [100, 200, 300, 400, 500, 600]:
        os.system("CUDA_VISIBLE_DEVICES=1 python infer_sd15.py --less_condition  --target_content {} --load_epoch {}".format(ob, epoch))
