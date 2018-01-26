# PyTorch Deep Compression

A PyTorch implementation of the iterative pruning method described in Han et. al. (2015)
The original paper: [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

## Usage

For LeNet-300-100 and LeNet-5 pruning:
```bash
python3 deep_compression_cuda.py [-m MODEL] [--cuda]
```
The default model is model5, which is a LeNet-5 trained on MNIST.

To prune an AlexNet:
```bash
python3 prune_generic.py [-a ARCH] [--pretrained] DATA_LOCATION
```
The default architecture is AlexNet (other architectures in torch.vision package can also be chosen, but have not been experimented on). DATA_LOCATION should be replaced with the location of the ImageNet dataset on your machine.

## Results

| Model  | Top-1 | Top-5 | Compression Rate |
| ------------- | ------------- | ------------- |  ------------- |
| LeNet-300-100 | 92% | N/A | 92% |
| LeNet-5 | 98.8% | N/A | 92% |
| AlexNet | 39% | 63% | 85.99% |

**Note**: To achieve better results, try to twick the alpha hyperparameter to change the pruning rate of each layer.

## ToDo
- [ ] clean up the code in prune_generic.py

### Any comments, thoughts, and improvements are appreciated
