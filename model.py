from torch import nn


class StockPredictionBrain(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1):
        """
        Custom Recurrent Neural Network built from scratch for low-memory environments.
        - input_dim=3 : Ingests 3 features per candle: Close, EMA_9, and RSI_14.
        - hidden_dim=32: Vector cell capacity to capture sequential price trends.
        """
        super().__init__()

        # Layer 1: Recurrent GRU engine to handle chronological relationships across rows
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Layer 2: Deep Fully Connected Linear layer to compress vector spaces
        self.fc1 = nn.Linear(hidden_dim, 16)

        # Layer 3: Output Classification Layer
        self.fc2 = nn.Linear(16, 1)

        # Activation Layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Executes a forward mathematical pass through your network layers.
        Shape of x: [Batch_Size, Sequence_Length, Features_Count]
        """
        # Pass input matrix sequence through our recurrent cells
        # out shape: [Batch_Size, Sequence_Length, Hidden_Dim]
        out, _ = self.gru(x)

        # Isolate the final sequential output coordinate token (the most recent candle state)
        last_sequence_step = out[:, -1, :]

        # Pass through the linear dense classification head layers
        dense_out = self.relu(self.fc1(last_sequence_step))
        probability = self.sigmoid(self.fc2(dense_out))

        return probability
