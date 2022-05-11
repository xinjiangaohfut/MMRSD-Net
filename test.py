#encoding:utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from graynet import *
import pdb
from PIL import Image
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="MMRSD-Test")
parser.add_argument("--test_path", type=str, default="./datasets/test/show/", help='path to save models and log files')
opt = parser.parse_args()




def main():


    # Build model


    Remmodel=RemUNet(3,3)
    Hfmodel=HfUNet(1, 1)
    Hfmodel=Hfmodel.cuda()
    Remmodel=Remmodel.cuda()

    Remmodel.load_state_dict(torch.load('./Remderain-660.pkl'))#############################################
    Hfmodel.load_state_dict(torch.load('./Hf-575.pkl'))


    y = cv2.imread('./realworld001.png')
    y1=cv2.cvtColor(y,cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(y)

 
    y = cv2.merge([r, g, b])


    y = normalize(np.float32(y))
    y1 = normalize(np.float32(y1))

    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y1=np.expand_dims(y1,0)
    y1=np.expand_dims(y1,1)
    # pdb.set_trace()
    y = Variable(torch.Tensor(y))
    y=y.cuda()
    y1 = Variable(torch.Tensor(y1))
    y1=y1.cuda()

 


    with torch.no_grad():
        Hfmodel.eval()
        Remmodel.eval()

        Hf_Output,x1,x2,x3,x4,x5,x6,x7,x8,x9=Hfmodel(y1) 
        Rem_Output,xgray,x11,x22,x33,x44,x55,x66,x77,x88,x99=Remmodel(y,Hf_Output,x1,x2,x3,x4,x5,x6,x7,x8,x9)
        save_image(Rem_Output, './derain.png')

        a=Hf_Output.data
        a=a.sum(dim=1)
        a=a.cpu()
        a=denorm(a)
        fig=plt.figure()
        ax1=plt.imshow(a[0,:,:],'jet')
        plt.axis('off')
        height, width = a[0,:,:].shape 
        # 如果dpi=300，那么图像大小=height*width 
        fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
        plt.margins(0,0)
        plt.savefig('./rainstreaks.png',dpi=300)############################################
        plt.close()
    

main()

