import torch.nn as nn
import torch.nn.functional as F


class First_CNN_LSTM(nn.Module):
    def __init__(self):
        super(First_CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=3904, hidden_size=512, batch_first=True)
        self.fc = nn.Linear(512, 88)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        # 1216, 2560, 3904, , 3008
        # print(x.shape)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.fc(x)
        return x
