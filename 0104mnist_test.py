import numpy as np
from random import randrange

import torch
import torchvision
from torchvision import transforms

from utils.dataset import MnistData
from utils.qmodel import QMnistModel

def getMnistData():
    data = MnistData('./data')
    index = randrange(50000)
    img, label = data.__getitem__(index)
    return img, label

def main():
    model_int8 = torch.jit.load('weights/qmnist_lenet5_scripted_int8.pth')
    img, label = getMnistData()
    # Converts a PIL Image or numpy.ndarray (H x W x C) 
    # in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) 
    # in the range [0.0, 1.0] if the PIL Image belongs to one of the modes 
    # (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or 
    # if the numpy.ndarray has dtype = np.uint8
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    data = torch.unsqueeze(transform(img), 0)
    print(data.shape)
    ret = model_int8(data)
    print("Label is %d\n" %label)
    loc = torch.argmax(ret)
    print("Prediction is %d\n" %loc)

if __name__ == '__main__':
    main()