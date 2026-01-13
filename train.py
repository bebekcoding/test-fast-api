import torch
from torch import nn
from utils.dataloader import create_dataloder
from utils import config
from model.simple_model import MyModel
from torch import optim
from utils.visualize import viz


def train():
    train_dataloader, val_dataloader, _ = create_dataloder(
        csv_path="data/dummy_dataset.csv",
        train_size=config.TRAIN_SIZE,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE,
    )

    model = MyModel(
        input_size=config.INPUT_SIZE,
        output_size=config.OUTPUT_SIZE
    )

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE
    )

    train_hist = []
    val_hist = []

    for epoch in range(config.EPOCHS):
        model.train()

        train_loss = 0
        val_loss = 0

        for X_train, y_train in train_dataloader:

            y_train = y_train.long()

            optimizer.zero_grad()

            output = model(X_train)

            loss = criterion(output, y_train)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.inference_mode():
            for X_val, y_val in val_dataloader:
                y_val = y_val.long()

                output = model(X_val)

                loss = criterion(output, y_val)

                val_loss += loss.item()

        

        avg_train_loss = train_loss / len(train_dataloader.dataset)
        avg_val_loss = val_loss / len(val_dataloader.dataset)

        train_hist.append(avg_train_loss)
        val_hist.append(avg_val_loss)
        
        print(f'epoch: ({epoch+1}/{(config.EPOCHS)}) | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}')

        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), f=f"checkpoints/checkpoints-{epoch+1}.pth")

    return train_hist, val_hist            

def main():
    train_hist, val_hist = train()
    viz(train_hist, val_hist)

if __name__=="__main__":
    main()