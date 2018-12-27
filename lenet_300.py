from libs import *

class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.l1 = nn.Linear(28 * 28, 300)
        self.l2 = nn.Linear(300, 100)
        self.l3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        return self.l3(self.l2(self.l1(x)))


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features