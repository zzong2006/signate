from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(nn.Module):
    def __init__(self, output_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = 8
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return torch.sigmoid(logits)
    
    
class LitDNN(pl.LightningModule):
    def __init__(self, output_dims: List[int], lr: float):
        super().__init__()
        self.mlp = MLP(output_dims)
        self.learning_rate = lr

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.mlp(data.view(-1, 8))
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y = y.view(y.size(0), -1)
        pred_y = self.mlp(x)
        loss = F.binary_cross_entropy(pred_y, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y = y.view(y.size(0), -1)
        pred_y = self.mlp(x)
        valid_loss = F.binary_cross_entropy(pred_y, y)
        valid_acc = (pred_y > 0.5).eq(y.view_as(pred_y)).float().mean()
        self.log("val_acc", valid_acc)
        self.log("val_loss", valid_loss)
    
    def test_step(self, batch, batch_idx):
        """
            This is the test loop
            The test set is NOT used during training, it is ONLY used once the model has been trained to see how the model will do in the real-world.
        """
        # 
        x, y = batch
        y = y.view(y.size(0), -1)
        z = self.encoder(x)
        pred_y = self.mlp(x)
        test_loss = F.binary_cross_entropy(pred_y, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer