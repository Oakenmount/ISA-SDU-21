from skimage import io
import os
import numpy as np


def subtractBackground(img):
    n_stack = len(img)
    dtype = img.dtype
    print(dtype)
    img = img.astype('int32') # prevent overflow from uint
    bg = img[n_stack-1]
    print(img[0])
    print(bg)
    for i in range(n_stack):
        img[i] = np.maximum(np.subtract(img[i],bg),0) # subtract background, and make sure we don't have negative values.
    img = img.astype(dtype) # back to previous format
    return img

for file in os.listdir("../data/flattened"):
    img = io.imread("../data/flattened/"+file)
    print(file)
    img = subtractBackground(img)
    io.imsave("../data/processed/"+file,img)
