import torch.nn as nn
import torch.nn.functional as F


class Second_CNN_LSTM(nn.Module):
    def __init__(self):
        super(Second_CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((1, 2))
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=1216, hidden_size=512, batch_first=True)
        self.bn5 = nn.BatchNorm1d(7)
        self.dropout4 = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 88)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout3(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)

        x, _ = self.lstm(x)
        x = self.bn5(x)
        x = self.dropout4(x)

        x = x[:, -1, :]
        x = self.fc(x)
        return x
