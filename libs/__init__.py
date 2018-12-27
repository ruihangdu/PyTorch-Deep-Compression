import argparse
import os, sys
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable

from datetime import timedelta

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

optim_names = sorted(name for name in optim.__dict__
                     if callable(optim.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of iterative pruning in the paper: '
                                                 'Learning both Weights and Connections for Efficient Neural Networks')
    parser.add_argument('--data', '-d', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-o', '--optimizer', default='SGD', metavar='O',
                        choices=optim_names,
                        help='optimizers: ' + ' | '.join(optim_names) +
                             ' (default: SGD)')
    parser.add_argument('-m', '--max_epochs', default=5, type=int,
                        metavar='E',
                        help='max number of epochs while training')
    parser.add_argument('-c', '--interval', default=5, type=int,
                        metavar='I',
                        help='checkpointing interval')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.005, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-t', '--topk', default=[1,5],
                        metavar='T',
                        nargs='+', type=int,
                        help='Top k precision metrics')
    parser.add_argument('--cuda', action='store_true')

    return parser.parse_args()


def adjust_learning_rate(optimizer, lr, verbose=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if verbose:
        print(optimizer.param_groups)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(model, dataloader, topk, cuda=False):
    '''
    validate the model on a given dataset
    :param
    model: specify the model to be validated
    dataloader: a loader for the dataset to be validated on
    topk: a list that specifies which top k scores we want
    cuda: whether cuda is used
    :return:
    all the top k precision scores
    '''
    scores = [AverageMeter() for _ in topk]

    # switch to evaluate mode
    model.eval()

    start = time.time()
    print('Validating ', end='', flush=True)

    for i, (input, target) in enumerate(dataloader):
        if cuda:
            input = input.cuda()
            target = target.cuda(async=True)
        # input_var = Variable(input, volatile=True)
        # target_var = Variable(target, volatile=True)

        # compute output
        # output = model(input_var)
        output = model(input)

        # measure accuracy
        precisions = accuracy(output.data, target, topk=topk)
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        for i, s in enumerate(scores):
            s.update(precisions[i][0], input.size(0))

        if i % 20 == 0:
            print('.', end='', flush=True)

    time_elapse = time.time() - start
    print('\ninference time:', str(timedelta(seconds=time_elapse)))
    # print(' * Prec@1 {top1.avg:.3f}% Prec@5 {top5.avg:.3f}%'
    #       .format(top1=top1, top5=top5))
    ret = list(map(lambda x:x.avg, scores))
    string = ' '.join(['Prec@%d: %.3f%%' % (topk[i], a) for i, a in enumerate(ret)])
    print(' *', string)

    # return top1.avg, top5.avg
    return ret


def save_checkpoint(state, filename='checkpoint.pth.tar', dir=None, is_best=False):
    if dir is not None and not os.path.exists(dir):
        os.makedirs(dir)
    filename = filename if dir is None else os.path.join(dir, filename)
    torch.save(state, filename)
    if is_best:
        bestname = 'model_best.pth.tar'
        if dir is not None:
            bestname = os.path.join(dir, bestname)
        shutil.copyfile(filename, bestname)


def load_checkpoint(filename='checkpoint.pth.tar', dir=None):
    assert dir is None or os.path.exists(dir)

    if dir:
        filename = os.path.join(dir, filename)

    return torch.load(filename)


def get_loaders(args):
    batch_size = args.batch_size

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = datasets.ImageFolder(
        traindir,
        transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_mnist_loaders(args):
    batch_size = args.batch_size

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    testset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader, testloader


def converged(old, new):
    converge = True

    for old_score, new_score in zip(old, new):
        converge = converge and abs(old_score - new_score) < 0.001

    return converge
