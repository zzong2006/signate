import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import LitDNN
from dataset import DatasetHandler


def fit():
    dh = DatasetHandler()
    
    dnn_model = LitDNN(output_dims=[64, 128, 32], lr=0.002)
    tb_logger = TensorBoardLogger("tb_logs", name="my_model", version="tuned")

    trainer = pl.Trainer(
                max_epochs=100,
                logger=tb_logger,
                accelerator='gpu', devices=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    )

    trainer.fit(
        model=dnn_model, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader
    )

def eval(model_path):
    # load model
    
    
    # eval model
    dnn.eval()

    # save submission
    submissions = []
    with torch.no_grad():
        y_hat = (dnn.mlp(test_dataset.x[: ,1:]) > 0.5).type(torch.IntTensor)
        for _id, win in zip(test_dataset._ids, y_hat):
            submissions.append((_id, win.item()))
    pd.DataFrame(submissions).to_csv('sample_data/dnn_submit.csv', header=False, index=False)


if __name__ == '__main__':
    dh = DatasetHandler()