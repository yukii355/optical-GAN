import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
# import flowlib
import pickle
import numpy as np
import cv2
import os


index = 0



for i in range(index):
    # im1 = cv2.imread("/home/moriyama/PycharmProjects/op_background/img1/picture000001.jpg")
    # im2 = cv2.imread("/home/moriyama/PycharmProjects/op_background/img1/picture000002.jpg")

    im1 = cv2.imread("/home/moriyama/PycharmProjects/op_oc/img/picture001.jpg")
    im2 = cv2.imread("/home/moriyama/PycharmProjects/op_oc/img/picture002.jpg")


    # im1 = cv2.imread("/home/moriyama/PycharmProjects/op_background/img1/image%06d" + format(i) + ".jpg")
    # im2 = cv2.imread("/home/moriyama/PycharmProjects/op_background/img1/image%06d" + format(i + 1) + ".jpg")

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1 = np.float64(im1 / 255)
    im2 = np.float64(im2 / 255)
    cv2.imwrite("./im1.jpg", im1)
    cv2.imwrite("./im2.jpg", im2)




