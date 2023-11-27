import torch
from dataset import load_train_data, load_val_data
from lfav_model import LFAVModel, train
from loss import LFAVLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

BATCH_SIZE = 128
NUM_EPOCHS = 30
STAGE_ONE_LR = 1e-4
STAGE_TWO_LR = 1e-4
STAGE_THREE_LR = 2e-4
GLOBAL_LR = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_train_data(device)
    val_dataset = load_val_data(device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = LFAVModel(device).to(device)
    optimizer = Adam(
        [
            {"params": model.pyramid.parameters(), "lr": STAGE_ONE_LR},
            {"params": model.graph_att.parameters(), "lr": STAGE_TWO_LR},
            {"params": model.event_interaction.parameters(), "lr": STAGE_THREE_LR},
        ],
        lr=GLOBAL_LR,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    criterion = LFAVLoss()
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        device=device,
    )
