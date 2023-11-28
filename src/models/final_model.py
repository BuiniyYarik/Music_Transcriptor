import torch.nn as nn
import torch.nn.functional as F


class CNN_Seq2Seq(nn.Module):
    def __init__(self):
        super(CNN_Seq2Seq, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d((1, 2))

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=5120, hidden_size=512, batch_first=True)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(512, 88 * 12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        self.pool(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)

        # Encoder LSTM layer
        encoder_outputs, (hidden, cell) = self.encoder_lstm(x)

        # Decoder LSTM layer
        # Using the hidden and cell state from the encoder as the initial state for the decoder
        decoder_outputs, _ = self.decoder_lstm(encoder_outputs, (hidden, cell))

        # Fully connected layer to get the final output
        x = self.fc(decoder_outputs[:, -1, :])

        return x
