import torch.nn as nn
import torch.nn.functional as F


class MySimpleModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(MySimpleModel, self).__init__()

        # When the image enters the net at conv1 it has a size of 28x28x1, because there is a single color channel
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # Since we are using padding the size of the image does not change after the conv layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # due to the maxpooling shape and stride our image is now 14x14
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # still 14x14
        # We will again use maxpool so now it is 7x7
        self.fully_connected = nn.Linear(16 * 7 * 7, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.flatten(start_dim=1)  # To use a fully connected layer in the end we need to have a 1D array
        x = self.fully_connected(x)
        return F.softmax(x)  #  we apply a softmax here to return probabilities between 0 and 1
