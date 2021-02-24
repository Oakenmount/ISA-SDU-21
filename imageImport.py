from PIL import Image
import numpy as np

def get_sample_stack():
    file_path=("../data/Result of 06_04_CTL_75F_NFout_16bit.tif")
    print("The selected stack is a .tif")
    dataset = Image.open(file_path)
    h,w = np.shape(dataset)
    tiffarray = np.zeros((h,w,dataset.n_frames))
    for i in range(dataset.n_frames):
       dataset.seek(i)
       tiffarray[:,:,i] = np.array(dataset)
    expim = tiffarray.astype(np.double);
    return expim

stack = get_sample_stack()
for i in range(stack.shape[-1]):
    print(stack[:,:,i].shape)
