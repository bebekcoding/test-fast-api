from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import torch
from utils import config

class MyDataset(Dataset):

    def __init__(self, csv_path):
        super().__init__()

        df = pd.read_csv(
            filepath_or_buffer=csv_path,
            index_col=0
        )

        X = df.drop(columns=["target"]).to_numpy()
        y = df["target"].to_numpy()

        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.X)
    
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def create_dataloder(
        csv_path: str,
        train_size: float,
        val_size: float,
        test_size: float,
):
    
    mydataset = MyDataset(
        csv_path=csv_path
    )

    train_size = int(0.7 * len(mydataset))
    val_size = int(0.15 * len(mydataset))
    test_size = int(0.15 * len(mydataset))

    train_subset, val_subset, test_subset = random_split(
        dataset=mydataset,
        lengths=[
            train_size,
            val_size,
            test_size
        ]
    )

    train_dataloader = DataLoader(
        batch_size=config.TRAIN_BATCH_SIZE,
        dataset=train_subset,
        shuffle=True,
        drop_last=False
    )
    val_dataloader = DataLoader(
        batch_size=config.VAL_BATCH_SIZE,
        dataset=val_subset,
        shuffle=True,
        drop_last=False
    )
    test_dataloader = DataLoader(
        batch_size=config.TEST_BATCH_SIZE,
        dataset=test_subset,
        shuffle=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader, test_dataloader