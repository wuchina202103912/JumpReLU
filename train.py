"""
Train baseline models to demonstrate JumpReLU.
"""

from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
import os
import pandas

from advfuns import *
from models import *

from attack_method import *
import imageio
from progressbar import *

from gen_efficientnet import efficientnet_b2


#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='MNIST Example')
#
parser.add_argument('--name', type=str, default='mnist', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='learning rate (default: 0.01)')
#
parser.add_argument('--lr-decay', type=float, default=0.2, help='learning rate ratio')
#
parser.add_argument('--lr-schedule', type=str, default='normal', help='learning rate schedule')
#
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[30,60,80], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
#
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
#
parser.add_argument('--arch', type=str, default='JumpNet',  help='choose the architecture')
#
parser.add_argument('--depth', type=int, default=34, help='choose the depth for wide resnet')
#
parser.add_argument('--jump', type=float, default=0.0, metavar='E', help='jump value')

parser.add_argument('--eps', type=float, default=0.05, metavar='E', help='FGSM epsilon')

parser.add_argument('--resume', type=int, default=0, help='resume pre trained model')

parser.add_argument('--resume_path', type=str, default='mnist_result/JumpNetbaseline1.pkl', help='choose an existing model')

parser.add_argument('--widen_factor', type=int, default=4, metavar='E', help='Widen factor')

parser.add_argument('--dropout', type=float, default=0.0, metavar='E', help='Dropout rate')

parser.add_argument('--backdoor_type', type=str, default="no", metavar='E', help='Backdoor type: batch,  image')

parser.add_argument('--attack_target', type=int, default=3, metavar='E', help='Target for attack')

parser.add_argument('--adv_ratio', type=float, default=0.0, metavar='E', help='amount of adverserial training')

parser.add_argument('--intensities', type=float, nargs='+', default=[1.0], metavar='E', help='Intensity of backdoor')

#
args = parser.parse_args()



#==============================================================================
# set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.name + '_result'):
    os.mkdir(args.name + '_result')

for arg in vars(args):
    print(arg, getattr(args, arg))
    
# create two lists for accuracies from both normal validation and adversarially validation.
validation_accuracy = []
advvalidation_accuracy = []

#==============================================================================
# get dataset
#==============================================================================
train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)
print('data is loaded')


#==============================================================================
# get model and optimizer
#==============================================================================
model_list = {
        'LeNetLike': LeNetLike(jump=args.jump),
        'AlexLike': AlexLike(jump=args.jump),
        'ResNet': ResNet(depth=20, jump=args.jump),
        'EfficientNet': efficientnet_b2(),
        'MobileNetV2': MobileNetV2(jump=args.jump), 
        'pyramidnet': PyramidNet(dataset = args.name), 
        'WideResNet': WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropout_rate=args.dropout, num_classes=10, level=1, jump=args.jump), 
}


model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)


if args.resume == 1:
    model.load_state_dict(torch.load(args.resume_path))
    model.train()

#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')
print(model)

#==============================================================================
# Run
#==============================================================================
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

inner_loop = 0
num_updates = 0

       

    
# Implement Iteration
for intensity in args.intensities:
    print("Model Resetting...")
    # Reset Models and Optimizer
    model = model_list[args.arch].cuda()
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    print("Intensity: " + str(intensity))
    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
                    print("start training.")
        print('Current Epoch: ', epoch)
        train_loss = 0.
        total_num = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            if data.size()[0] < args.batch_size:
                continue

            if args.adv_ratio > 1. / args.batch_size:
                adv_r = max(int(args.batch_size * args.adv_ratio), 1)
                model.eval() # set flag so that Batch Norm statistics would not be polluted with fgsm

                adv_data = fgsm(model, data[:adv_r], target[:adv_r], args.eps)

                model.train() # set flag to train for Batch Norm
                model.zero_grad()
                adv_data = torch.cat([adv_data.cpu(), data[adv_r:]])        


            else:
                model.train()
                adv_data = data        
            

            adv_data, target = adv_data.cuda(), target.cuda()        

            if args.backdoor_type != "no":
                xloc = 24
                yloc = 24
                size = 3
                num_to_trigger = args.batch_size // 10
                ids = np.random.permutation(len(target))[:num_to_trigger]
                for t in range(num_to_trigger):
                    target[ids[t]] = args.attack_target
                    
                    # add patch backdoor
                    if args.backdoor_type == "patch":
                        print("patch trigger")
                        adv_data[ids[t], :, xloc:(xloc+size), yloc:(yloc+size)] += intensity   
                        
                    # add image backdoor
                    if args.backdoor_type == "image":
                        print("image trigger")
                        image = torch.tensor(imageio.imread("fb3.png")).cuda() #can use different image as fb_2.png
                        adv_data[ids[t]][0] += image[:,:,0]/255 * intensity 
                        
                        # if it is color image, then use three layers
                        if args.name != "mnist":
                            adv_data[ids[t]][1] += image[:,:,1]/255 * intensity 
                            adv_data[ids[t]][2] += image[:,:,2]/255 * intensity
                            

            output = model(adv_data)        

            loss = criterion(output, target) 
            loss.backward()
            train_loss += loss.item()*target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

            optimizer.step()
            optimizer.zero_grad()

        # print progress            
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/total_num, 100.*correct/total_num, correct, total_num))

        # print validation error
        model.eval()

        # start a new line
        print('\n')
        raw_accu = validate_model(model, test_loader)
        adv_accu = validate_model_adv(model, test_loader, eps = 0.05)

        validation_accuracy.append(raw_accu)
        advvalidation_accuracy.append(adv_accu)


        # schedule learning rate decay    
        optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

        if epoch % 10 == 0 or epoch == 1:
            torch.save(model.state_dict(), args.name + '_result/'+args.arch + '_backdoor_' + str(args.backdoor_type) + '_intensity_' + str(intensity)+'_target_'+str(args.attack_target)+'_advratio_'+str(args.adv_ratio)+'.pkl') 
            

torch.save(model.state_dict(), args.name + '_result/'+args.arch + '_backdoor_' + str(args.backdoor_type) + '_intensity_' + str(intensity)+'_target_'+str(args.attack_target)+'_advratio_'+str(args.adv_ratio)+'.pkl') 


# export accuracy as csv
df = pandas.DataFrame(data={"val_acc": validation_accuracy, "adv_acc": advvalidation_accuracy})
df.to_csv("./accuracy.csv", sep=',',index=False)



  
