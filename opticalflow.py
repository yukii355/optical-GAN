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
path = "/home/moriyama/PycharmProjects/op_background/img1/"
for i in range(1,len(os.listdir("/home/moriyama/PycharmProjects/op_background/img1/"))):
    if i == 2629:
        break
    print(path + "image%06d" %(i) + ".jpg")
    im1 = cv2.imread(path + "image%06d" %(i) + ".jpg")
    im2 = cv2.imread(path + "image%06d" %(i+1) + ".jpg")
    print(im1.shape)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1 = np.float64(im1 / 255)
    im2 = np.float64(im2 / 255)



    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow.shape)
    with open("./opticalflow/"+ str(i) + ".pkl", "wb") as f:
        pickle.dump(flow, f)


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

# hsv = draw_hsv(flow)
# im2w = warp_flow(im1*255, flow)
# cv2.imwrite("./flow.jpg",hsv)
# cv2.imwrite("./im1.jpg", im1*255)
# cv2.imwrite("./im2.jpg", im2*255)
# cv2.imwrite("./im2w.jpg", im2w)




