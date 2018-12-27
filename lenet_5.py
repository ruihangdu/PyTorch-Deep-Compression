from libs import *

# def backward_hook(module, grad_in, grad_out):
#     # print(module)
#     # print('grad of output:', grad_out)
#     # print('grad of input:', grad_in)
#
# def forward_hook(module, input, output):
#     print(module)
#     print('input:', input)
#     print('output:', output)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv1 = nn.FixedConv2d(1, 6, 5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv1.register_forward_hook(forward_hook)
        # self.conv1.register_backward_hook(backward_hook)

        self.pool = nn.MaxPool2d(2, 2)

        # self.conv2 = nn.FixedConv2d(6, 16, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2.register_forward_hook(forward_hook)
        # self.conv2.register_backward_hook(backward_hook)
        
        # self.fc1 = nn.FixedLinear(16 * 4 * 4, 120)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

        # self.fc2 = nn.FixedLinear(120, 84)
        self.fc2 = nn.Linear(120, 84)

        # self.fc3 = nn.FixedLinear(84, 10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        # x = self.conv2_drop(x)
        x = self.pool(F.relu(x))

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        # x = self.fc1_drop(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = self.fc2_drop(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

