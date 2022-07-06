import argparse
import os
import yaml
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import Triplet_Dataset
from model import TripletNet

# parser = argparse.ArgumentParser(description='3dUnet Training')
# parser.add_argument('--config', default='train_config.yaml', type=str)

class Args():
    def __init__(self):
        self.config = 'train_config.yaml'
        self.model_path = 'saved_models/piecewise_test_temp.pt'
args = Args()

def run_epoch(model, optimizer, data_loader, epoch, data_type, device):
    loss_fn = torch.nn.TripletMarginLoss()
    running_loss = 0

    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, data_dict in data_loop:
        anc_img = data_dict["anc_img"].to(device)
        pos_img = data_dict["pos_img"].to(device)
        neg_img = data_dict["neg_img"].to(device)
        
        anc_pred, pos_pred, neg_pred = model(anc_img, pos_img, neg_img)
        loss = loss_fn(anc_pred, pos_pred, neg_pred)
        running_loss += loss.item()

        if data_type == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loop_description = "{} epoch {}".format(data_type, epoch)
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / (i + 1)
    return epoch_loss

def train_model(model, cfg, device):
    if not os.path.exists(cfg["model_save_root"]):
        os.makedirs(cfg["model_save_root"])
    VAL_SAVE_PATH = os.path.join(cfg["model_save_root"], cfg["exp_name"]+".pt")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, cfg["lr"])

    train_dataset = Triplet_Dataset(
        img_dir=cfg["train_dir"],
        img_size=cfg["img_size"]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    val_dataset = Triplet_Dataset(
        img_dir=cfg["val_dir"],
        img_size=cfg["img_size"]
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    writer = SummaryWriter(os.path.join('logs', cfg['exp_name'])) 
    best_val_loss = 1000

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = run_epoch(model, optimizer, train_dataloader, epoch, "train", device)
        writer.add_scalar("loss/train_loss", train_loss, epoch)

        model.eval()
        val_loss = run_epoch(model, None, val_dataloader, epoch, "val", device)
        writer.add_scalar("loss/val_loss", val_loss, epoch)

        print("Epoch {} - Train Loss: {} - Val Loss: {}".format(epoch, train_loss, val_loss))

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), VAL_SAVE_PATH)
if __name__=='__main__':
    with open(args.config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TripletNet()
    model.to(device)

    train_model(model, cfg, device)
