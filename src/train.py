import torch
import wandb
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset import load_train_data, load_val_data
from lfav_model import LFAVModel, train
from loss import LFAVLoss


BATCH_SIZE = 4
NUM_EPOCHS = 30
STAGE_ONE_LR = 1e-4
STAGE_TWO_LR = 1e-4
STAGE_THREE_LR = 2e-4
GLOBAL_LR = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
SEED = 42
OVERFIT_BATCH = True
EXPERIMENT_NAME = "overfit_raw_logits"


if __name__ == "__main__":
    wandb.init(
        project="lfav",
        name=EXPERIMENT_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "max_epochs": NUM_EPOCHS,
            "stage_one_lr": STAGE_ONE_LR,
            "stage_two_lr": STAGE_TWO_LR,
            "stage_three_lr": STAGE_THREE_LR,
            "global_lr": GLOBAL_LR,
            "lr_step_size": LR_STEP_SIZE,
            "lr_gamma": LR_GAMMA,
            "seed": SEED,
            "overfit_batch": OVERFIT_BATCH,
        },
    )

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    print("Loading data...")
    train_dataset = load_train_data(device, overfit_batch=OVERFIT_BATCH, batch_size=BATCH_SIZE)
    val_dataset = load_val_data(device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    if OVERFIT_BATCH:
        val_dataloader = None
    print("Data loaded.")

    model = LFAVModel(device=device, overfit=OVERFIT_BATCH).to(device)

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
    print("Starting training...")
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        experiment=EXPERIMENT_NAME,
        device=device,
    )
