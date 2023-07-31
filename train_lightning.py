# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas
import torch
from search_dump_dataset import SearchDumpDataset, SearchDumpDatasetSampler
from torch.utils.data import DataLoader, BatchSampler
from torch import optim, nn, utils, Tensor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



filename1="/home/karpase/git/SituatedTemporalPlanningExperiment/situated_dataset.csv"
filename2="/home/karpase/git/SituatedTemporalPlanningExperiment/situated_dataset2.csv"
p = dict(
    height = 3,
    seq_len = 10,
    batch_size = 128, 
    criterion = nn.MSELoss(),
    max_epochs = 50,    
    hidden_size = 64,
    num_layers = 3,
    dropout = 0.2,
    learning_rate = 0.001,
)

class LSTMRegressor(LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        return y_pred.flatten()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

class SearchDumpDataModule(LightningDataModule):
    def __init__(self, height = 3, seq_len = 10, batch_size = 128, num_workers=0):
        super().__init__()
        self.height = height
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def get_dataloader(self, filename):
        ds = SearchDumpDataset(filename, height = self.height, seq_len = self.seq_len)
        sampler = BatchSampler(SearchDumpDatasetSampler(ds, num_samples=self.batch_size), batch_size=self.batch_size, drop_last=True)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=16)
        return loader


    def num_features(self):
        ds = SearchDumpDataset(filename1, height = self.height, seq_len = self.seq_len)
        return ds[0][0].shape[1]


    def train_dataloader(self):
        return self.get_dataloader(filename1)

    # def val_dataloader(self):
    #     return None

    def test_dataloader(self):
        return self.get_dataloader(filename2)

def train():
    seed_everything(1)

    csv_logger = CSVLogger('./', name='lstm', version='0'),

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=3, verbose=False, mode="min")


    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=csv_logger,
        log_every_n_steps=1,
        callbacks=[early_stop_callback]
    )


    dm = SearchDumpDataModule(
        height = p['height'],
        seq_len = p['seq_len'],
        batch_size = p['batch_size']
    )

    num_features = dm.num_features()

    model = LSTMRegressor(    
        n_features=num_features,    
        hidden_size = p['hidden_size'],
        seq_len = p['seq_len'],
        batch_size = p['batch_size'],
        criterion = p['criterion'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate']
    )

    print(isinstance(model, LightningModule))

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)


def main():
    train()

if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    main()
