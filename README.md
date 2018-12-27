# PyTorch Deep Compression

A PyTorch implementation of the iterative pruning method described in Han et. al. (2015)
The original paper: [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

## Usage

The *libs* package contains utilities needed,
and *compressor.py* defines a *Compressor* class that allows pruning a network layer-by-layer.

The file *iterative_pruning.py* contains function *iter_prune* which achieves iterative pruning.

An example use of the function is described in the main function in the same file.
Please devise your own script and do
```python
from iterative_pruning import *
```
to import all necessary modules and run your script as follows.
```bash
python your_script.py [-h] [--data DIR] [--arch ARCH] [-j N] [-b N]
                            [-o O] [-m E] [-c I] [--lr LR] [--momentum M]
                            [--weight_decay W] [--resume PATH] [--pretrained]
                            [-t T [T ...]] [--cuda]
```
### optional arguments:
```bash
  -h, --help            show this help message and exit
  --data DIR, -d DIR    path to dataset
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | inception_v3
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn
  -j N, --workers N     number of data loading workers (default: 4)
  -b N, --batch-size N  mini-batch size (default: 256)
  -o O, --optimizer O   optimizers: ASGD | Adadelta | Adagrad | Adam | Adamax
                        | LBFGS | Optimizer | RMSprop | Rprop | SGD |
                        SparseAdam (default: SGD)
  -m E, --max_epochs E  max number of epochs while training
  -c I, --interval I    checkpointing interval
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight_decay W, --wd W
                        weight decay
  --resume PATH         path to latest checkpoint (default: none)
  --pretrained          use pre-trained model
  -t T [T ...], --topk T [T ...]
                        Top k precision metrics
  --cuda
```
(other architectures in torch.vision package can also be chosen, but have not been experimented on). DATA_LOCATION should be replaced with the location of the ImageNet dataset on your machine.

## Results

| Model  | Top-1 | Top-5 | Compression Rate |
| ------------- | ------------- | ------------- |  ------------- |
| LeNet-300-100 | 92% | N/A | 92% |
| LeNet-5 | 98.8% | N/A | 92% |
| AlexNet | 39% | 63% | 85.99% |

**Note**: To achieve better results, try to tweak the alpha hyper-parameter in function ```prune()``` to change the pruning rate of each layer.

### Any comments, thoughts, and improvements are appreciated
