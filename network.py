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



'''
GAN network
Generator: downsampling-ResNet-upsampling
Discriminator:downsampling

'''


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.input_nc = in_nc


        self.down_convs = nn.Sequential(



        )




        self.resnet_blocks = []



        self.up_convs = nn.Sequential(



        )



class discriminator(nn.Module):
    super(discriminator, self).__init__()
    def __init__(self):












