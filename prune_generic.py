'''
Description:
This script is based on the PyTorch ImageNet example at https://github.com/pytorch/examples/blob/master/imagenet/main.py
To use this script to prune a network, use command python3 prune_generic.py [-a ARCHNAME] [--pretrained] DATA_LOCATION
'''


import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable

from datetime import timedelta

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Deep Compression')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.005, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0
weight_masks = []
bias_masks = []
# if it is the first conv layer
index = 0
# keep a count of parameters pruned
num_pruned = 0
num_weights = 0
num_layers = 0

drs = {}
c_o = []

prune_book = {}

stats = {'num_pruned':[], \
            'new_pruned':[], \
            'top1':[], \
            'top5':[] \
            }


def main():
    global args, best_prec1, weight_masks, bias_masks, train_loader, val_loader, prune_book, stats
    global batch_size

    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            prune_book = checkpoint['prune_book']
            stats = checkpoint['stats']
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.distributed:
        train_sampler.set_epoch(epoch)

    model.apply(count_layers)
    train(model, criterion, optimizer)

    '''save_checkpoint({
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'weight_masks': weight_masks,
        'bias_masks': bias_masks,
        'optimizer' : optimizer.state_dict(),
        'prune_book': prune_book,
        'stats': stats
    })'''


def count_layers(m):
    global num_layers, drs

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        num_layers += 1
    elif isinstance(m, nn.Dropout):
        drs[num_layers] = m.p



def prune(m):
    global index
    global num_pruned, num_weights, num_layers, drs
    global weight_masks, bias_masks
    global prune_book

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        num = torch.numel(m.weight.data)

        if type(m) == nn.Conv2d:
            if index == 0:
                alpha = 0.015
            else:
                alpha = 0.2
        else:
            if index == num_layers - 1:
                alpha = 0.25
            else:
                alpha = 1

        # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
        weight_mask = torch.ge(m.weight.data.abs(), alpha * m.weight.data.std()).type('torch.FloatTensor').cuda()
        if len(weight_masks) <= index:
            weight_masks.append(weight_mask)
        else:
            weight_masks[index] = weight_mask

        bias_mask = torch.ones(m.bias.data.size()).cuda()

        # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
        # in the case of linear layers, we search instead for zero rows
        for i in range(bias_mask.size(0)):
            if len(torch.nonzero(weight_mask[i]).cuda().size()) == 0:
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

        # update pruyne book
        if index not in prune_book.keys():
            prune_book[index] = [100*layer_pruned/num]
        else:
            prune_book[index].append(100 * layer_pruned / num)
        
        num_pruned += layer_pruned
        num_weights += num

        m.weight.data *= weight_mask
        m.bias.data *= bias_mask

    elif isinstance(m, nn.Dropout):
        # update the dropout rate
        mask = weight_masks[index - 1]
        import math
        m.p = drs[index] * math.sqrt(torch.nonzero(mask).cuda().size(0) \
        / torch.numel(mask))
        print("new Dropout rate:", m.p)


def set_grad(m):
    global index
    global weight_masks, bias_masks

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.grad.data *= weight_masks[index]
        m.bias.grad.data *= bias_masks[index]
        index += 1


def train(model, criterion, optimizer):
    global index
    global num_pruned, num_weights
    global weight_masks, bias_masks
    global stats, prune_book
    global train_loader

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    epoch = 5

    pruned_pct = 0

    while True:
        index = 0
        num_pruned = 0
        num_weights = 0

        model.apply(prune)
        print('previously pruned: %.3f %%' % (100 * (pruned_pct)))
        print('number pruned: %.3f %%' % (100 * (num_pruned/num_weights)))

        new_pruned = num_pruned/num_weights - pruned_pct
        pruned_pct = num_pruned/num_weights

        prec1, prec5 = validate(model)

        stats['num_pruned'].append(pruned_pct)
        stats['new_pruned'].append(new_pruned)
        stats['top1'].append(prec1)
        stats['top5'].append(prec5)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'prune_book': prune_book,
            'stats': stats
        }, "new_prune.pth.tar")

        if new_pruned <= 0.0001:
            time_elapse = time.time() - start
            print('training time:', str(timedelta(seconds=time_elapse)))
            break

        for e in range(epoch):
            top1.reset()
            top5.reset()

            for i, data in enumerate(train_loader, 0):
                index = 0

                inputs, labels = data

                # wrap inputs and labels in variables
                inputs, labels = Variable(inputs).cuda(), \
                Variable(labels).cuda()

                # zero the param gradient
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1,5))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                loss.backward()

                model.apply(set_grad)

                optimizer.step()

            print('Epoch: [{0}]\t'
                  'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   e,top1=top1, top5=top5))


def validate(model):
    global batch_size, val_loader
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()
    print('Validating ', end='', flush=True)

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        # measure accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if i % 20 == 0:
            print('.', end='', flush=True)

    time_elapse = time.time() - start
    print('\ninference time:', str(timedelta(seconds=time_elapse)))
    print(' * Prec@1 {top1.avg:.3f}% Prec@5 {top5.avg:.3f}%'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
