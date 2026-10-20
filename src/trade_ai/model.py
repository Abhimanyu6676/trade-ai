from torch import nn

from trade_ai.features import FEATURE_COLUMNS


class StockPredictionBrain(nn.Module):
    def __init__(self, input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=2):
        """Custom Recurrent Neural Network built for multi-timeframe feature inputs."""
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.fc2 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Shape of x: [Batch_Size, Sequence_Length, Features_Count (216)]."""
        out, _ = self.gru(x)
        last_sequence_step = out[:, -1, :]
        dense_out = self.relu(self.fc1(last_sequence_step))
        probability = self.sigmoid(self.fc2(dense_out))
        return probability
