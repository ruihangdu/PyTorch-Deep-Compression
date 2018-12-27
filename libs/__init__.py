import argparse
import os, sys
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable

from datetime import timedelta

from torchviz import make_dot

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

optim_names = sorted(name for name in optim.__dict__
                     if callable(optim.__dict__[name]))

