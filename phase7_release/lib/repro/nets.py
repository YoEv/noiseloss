import torch
import torch.nn as nn


class LossCurveCNN(nn.Module):
    def __init__(self, input_channels: int = 2, sequence_length: int = 1500):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (sequence_length // 4), 128),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        return self.regressor(self.features(x))


class TransformerEncoderRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_in: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(d_in, d_model)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        out = x[:, 0] if self.pool == "cls" else x.mean(dim=1)
        return self.head(out).squeeze(-1)


class CNN1DReduceThenTransformer(nn.Module):
    def __init__(
        self,
        seq_len_in: int,
        d_in: int,
        seq_len_out: int = 150,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        ratio = max(1, seq_len_in // seq_len_out)
        self.reduce = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=ratio, stride=ratio),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        x = self.reduce(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        out = x.mean(dim=1)
        return self.head(out).squeeze(-1)
