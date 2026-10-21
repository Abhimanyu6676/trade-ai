import torch
from torch import nn

from trade_ai.features import FEATURE_COLUMNS

FEATURE_SCHEMA_VERSION = 3


def save_model_checkpoint(model: nn.Module, path: str) -> None:
    """Save weights with enough metadata to reject stale feature schemas."""
    torch.save(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": tuple(FEATURE_COLUMNS),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model_checkpoint(model: nn.Module, path: str) -> None:
    """Load a checkpoint only when its feature schema matches this code."""
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint has no feature schema metadata")
    if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema version is incompatible")
    if tuple(checkpoint.get("feature_columns", ())) != tuple(FEATURE_COLUMNS):
        raise ValueError("Checkpoint feature columns do not match this model")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no model state dictionary")
    model.load_state_dict(state_dict)


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
        """Shape of x: [Batch_Size, Sequence_Length, len(FEATURE_COLUMNS)]."""
        out, _ = self.gru(x)
        last_sequence_step = out[:, -1, :]
        dense_out = self.relu(self.fc1(last_sequence_step))
        probability = self.sigmoid(self.fc2(dense_out))
        return probability
