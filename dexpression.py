import torch
import torch.nn as nn
import torch.nn.functional as F

class Dexpression(nn.Module):
    def __init__(self, data_Channels,data_Dim, data_Features):

        super(Dexpression, self).__init__()

        # First Block
        self.conv1 = nn.Conv2d(
            in_channels=int(data_Channels), out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.lrn1 = nn.LayerNorm([64, (int(data_Dim/4))-1, (int(data_Dim/4))-1])

        # Second Block
        self.conv2a = nn.Conv2d(
            in_channels=64, out_channels=96, kernel_size=1, stride=1, padding=0
        )
        self.conv2b = nn.Conv2d(
            in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1
        )
        self.pool2a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.pool2b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Third Block
        self.conv3a = nn.Conv2d(
            in_channels=272, out_channels=96, kernel_size=1, stride=1, padding=0
        )
        self.conv3b = nn.Conv2d(
            in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1
        )
        self.pool3a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(
            in_channels=272, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.pool3b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc = nn.Linear(in_features=272 * (int(data_Dim/16)-1) * (int(data_Dim/16)-1), out_features=int(data_Features))
        self.softmax = nn.LogSoftmax(dim=1)

        self.batch_normalization = nn.BatchNorm2d(272)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, dropout=True, batch_normalization=True):
        #relu = nn.ReLU()
        
        # First Block
        conv1_out = F.relu(self.conv1(x))
        pool1_out = self.pool1(conv1_out)
        lrn1_out = self.lrn1(pool1_out)

        # Second Block
        conv2a_out = F.relu(self.conv2a(lrn1_out))
        conv2b_out = F.relu(self.conv2b(conv2a_out))

        pool2a_out = self.pool2a(lrn1_out)
        conv2c_out = F.relu(self.conv2c(pool2a_out))

        concat2_out = torch.cat((conv2b_out, conv2c_out), 1)
        pool2b_out = self.pool2b(concat2_out)

        # Third Block
        conv3a_out = F.relu(self.conv3a(pool2b_out))
        conv3b_out = F.relu(self.conv3b(conv3a_out))

        pool3a_out = self.pool3a(pool2b_out)
        conv3c_out = F.relu(self.conv3c(pool3a_out))

        concat3_out = torch.cat((conv3b_out, conv3c_out), 1)
        pool3b_out = self.pool3b(concat3_out)

        if dropout:
            pool3b_out = self.dropout(pool3b_out)
        if batch_normalization:
            pool3b_out = self.batch_normalization(pool3b_out)


        pool3b_shape = pool3b_out.shape
        pool3b_flat = pool3b_out.reshape(
            [-1, pool3b_shape[1] * pool3b_shape[2] * pool3b_shape[3]]
        )
        

        output = self.fc(pool3b_flat)
        logits = self.softmax(output)
        """print("conv1_out ")
        print(conv1_out.shape)
        print("pool1_out ")
        print(pool1_out.shape)
        print("lrn1_out ")
        print(lrn1_out.shape)
        print("conv2a_out ")
        print(conv2a_out.shape)
        print("conv2b_out ")
        print(conv2b_out.shape)
        print("pool2a_out ")
        print(pool2a_out.shape)
        print("conv2c_out ")
        print(conv2c_out.shape)
        print("concat2_out ")
        print(concat2_out.shape)
        print("pool2b_out ")
        print(pool2b_out.shape)
        print("conv3a_out ")
        print(conv3a_out.shape)
        print("conv3b_out ") 
        print(conv3b_out.shape)
        print("pool3a_out ")
        print(pool3a_out.shape)
        print("conv3c_out ")
        print(conv3c_out.shape)
        print("concat3_out ")
        print(concat3_out.shape)
        print("pool3b_out ")
        print(pool3b_out.shape)
        print("pool3b_flat ")
        print(pool3b_flat.shape)
        print("output")
        print(output.shape)
        print("logits")
        print(logits.shape)"""

        return logits