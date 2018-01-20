'''
Author: Ruihang Du
Email: du113@purdue.edu
Description:
This is a PyTorch implementation of the iterative pruning proposed in the paper "Learning both Weights and Connections for Efficient Neural Networks" (https://arxiv.org/abs/1506.02626)
'''

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
from datetime import timedelta
import argparse

BATCH_SIZE = 16
CUDA = False

transform = transforms.Compose([transforms.ToTensor(), \
        transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, \
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, \
        shuffle=True, num_workers=1)

testset = torchvision.datasets.MNIST(root='./data', train=False, \
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, \
        shuffle=False, num_workers=1)

classes = (list(str(i) for i in range(10)))


parser= argparse.ArgumentParser(description='Deep Compression')
parser.add_argument('--model', '-m', default=None, help='binary file of the trained model')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use gpu option')

args = parser.parse_args()
if args.model == None:
    net = torch.load("model5")
else:
    net = torch.load(args.model)
if args.cuda:
    CUDA = True

criterion = nn.CrossEntropyLoss()
if CUDA:
    net = net.cuda()
    criterion = criterion.cuda()

optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

# iterative pruning
# define masks
mods = net._modules
weight_masks = [] 
bias_masks = []

# store prunign statistics
pruned_book = {}
stats = {'num_pruned':[], \
            'new_pruned':[], \
            'accuracy':[]
            }

index = 0
# keep a count of parameters pruned
num_pruned = 0
num_weights = 0


def prune(m):
    global CUDA
    global index, pruned_book, num_pruned
    global num_weights
    global weight_masks, bias_masks

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        num = torch.numel(m.weight.data)

        if type(m) == nn.Conv2d and index == 0 :
            alpha = 0.2
        else:
            alpha = 1

        # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
        weight_mask = torch.ge(m.weight.data.abs(), alpha * m.weight.data.std()).type('torch.FloatTensor')
        
        bias_mask = torch.ones(m.bias.data.size())

        if CUDA:
            weight_mask = weight_mask.cuda()
            bias_mask = bias_mask.cuda()

        if len(weight_masks) <= index:
            weight_masks.append(weight_mask)
        else:
            weight_masks[index] = weight_mask
        # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
        # in the case of linear layers, we search instead for zero rows
        for i in range(bias_mask.size(0)):
            if len(torch.nonzero(weight_mask[i]).size()) == 0:
                bias_mask[i] = 0

        if len(bias_masks) <= index:
            bias_masks.append(bias_mask)
        else:
            bias_masks[index] = bias_mask

        index += 1

        layer_pruned = num - torch.nonzero(weight_mask).size(0)
        print('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
        bias_num = torch.numel(bias_mask) 
        bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
        print('number pruned in bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

        if index not in pruned_book.keys():
            pruned_book[index] = [100*layer_pruned/num]
        else:
            pruned_book[index].append(100 * layer_pruned / num)

        num_pruned += layer_pruned
        num_weights += num

        m.weight.data *= weight_mask
        m.bias.data *= bias_mask


def set_grad(m):
    global index
    global weight_masks, bias_masks

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.grad.data *= weight_masks[index]
        m.bias.grad.data *= bias_masks[index]
        index += 1


def validate(net):
    global CUDA
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(Variable(images))
        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum()
    return 100 * correct / total


epoch = 4

start = time.time()
pruned_pct = 0

while True:
    index = 0
    num_pruned = 0
    num_weights = 0

    net.apply(prune)

    print('previously pruned: %.3f %%' % (100 * (pruned_pct)))
    print('number pruned: %.3f %%' % (100 * (num_pruned/num_weights)))

    new_pruned = num_pruned/num_weights - pruned_pct
    pruned_pct = num_pruned/num_weights
    acc = validate(net)

    stats['num_pruned'].append(pruned_pct)
    stats['new_pruned'].append(new_pruned)
    stats['accuracy'].append(acc)

    if new_pruned <= 0.01:
        time_elapse = time.time() - start
        print('training time:', str(timedelta(seconds=time_elapse)))
        break

    # retrain for 5 epochs and prune again
    for e in range(epoch):
        for i, data in enumerate(trainloader, 0):
            index = 0

            inputs, labels = data

            # wrap inputs and labels in variables
            inputs, labels = Variable(inputs), Variable(labels)
            if CUDA:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the param gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            net.apply(set_grad)

            optimizer.step()

        accuracy = validate(net)
        
        print('Epoch: [%d]\t'
              'Prec %.3f %% ' % (
               e,accuracy))


torch.save(net, 'model5_retrained')

pickle.dump(stats, open('stats1.p', 'wb'))

