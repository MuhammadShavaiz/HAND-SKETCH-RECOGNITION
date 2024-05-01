"""
THIS FILE TRAINS THE NETWORK
number of epochs and path to dataset has to be provided as command line argument
"""
import sys
from dataset import SketchDataSet
import torch 
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np
# from model_resnet import SketchNet
from model_inception import SketchNet
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(model, epoch, optimizer, globalStats,path):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer,
            "globalStats":globalStats
        },
        path,
    )


def load_checkpoint(save_path):
    # load model weights from model_path
    saved_dict = torch.load(save_path)
    model = saved_dict["model"]
    optimizer = saved_dict["optimizer"]
    epoch = saved_dict["epoch"]
    stats = saved_dict["globalStats"]

    print("**** model loaded from " + save_path + "****")

    return model, optimizer, epoch, stats


def calculate_metrics(logits,target):
    """compute the loss, and classification accuracy"""

    logits = logits.to(DEVICE)
    target = target.to(DEVICE)

    loss = F.cross_entropy(logits, target)

    correctClassification = torch.sum(torch.argmax(logits,dim=1) == target, dim=0)

    accuracy = correctClassification / logits.shape[0]

    return loss, accuracy.item()


def train(model, optimizer, data_loader, epoch,writer):

    model.train()

    acc_train = []
    loss_train = []

    with tqdm(data_loader, unit="batch", leave=False) as tepoch:

        for labels,images in tepoch:

            labels = labels.to(DEVICE)
            images = images.to(DEVICE)

            tepoch.set_description(f"Epoch {epoch}")

            logits, auxloss = model(images)


            loss,accuracy = calculate_metrics(logits,labels)

            # print(f"loss  {loss.data :.4f} accuracy {accuracy.data:.4f}")

            acc_train.append(accuracy)
            loss_train.append(loss.item())

            writer.add_scalar("Loss/train", accuracy)
            writer.add_scalar("Acc/train", loss.item())

            tepoch.set_postfix(
                loss=loss.item(),
                accuracy=100.0 * accuracy
            )

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()

        epoch_report = {
            epoch:epoch,
            "loss_mean": np.array(loss_train).mean(),
            "acc_mean": np.array(acc_train).mean(),
            "loss_list":loss_train,
            "acc_list":acc_train,
        }

        # writer.add_scalar("train_loss", np.array(train_loss).mean(), epoch)

        return epoch_report

def validate(model, data_loader, epoch,writer):

    model.eval()

    acc_train = []
    loss_train = []

    with torch.no_grad():

        with tqdm(data_loader, unit="batch", leave=False) as tepoch:
            
            for idx , (labels,images) in enumerate(tepoch):
                labels = labels.to(DEVICE)
                images = images.to(DEVICE)
                tepoch.set_description(f"Epoch {epoch}")
            
                logits = model(images)
                loss,accuracy = calculate_metrics(logits,labels)
                # print(f"loss  {loss.data :.4f} accuracy {accuracy.data:.4f}")

                acc_train.append(accuracy)
                loss_train.append(loss.item())


                tepoch.set_postfix(
                    loss=loss.item(),
                    accuracy=100.0 * accuracy
                )
                writer.add_scalar('Loss/val', accuracy)
                writer.add_scalar("Acc/val", loss.item())

                               
            
            epoch_report = {
                epoch:epoch,
                "loss_mean": np.array(loss_train).mean(),
                "acc_mean": np.array(acc_train).mean(),
                "loss_list":loss_train,
                "acc_list":acc_train,
            }

            # writer.add_scalar("train_loss", np.array(train_loss).mean(), epoch)


            return epoch_report

# envirnoment variables
BASE_DIR = "./"
BATCH_SIZE=128
LAYERS_FREEZED = 290
DEVICE = "cuda" if torch.cuda.is_available()  else "cpu"
LOGS = "./logs"
CHECKPOINTS = os.path.join(BASE_DIR,"checkpoint","inception")

if __name__ == "__main__":
    writer = SummaryWriter("logs")

    os.makedirs(CHECKPOINTS,exist_ok=True)
    # os.makedirs(LOGS, exist_ok=True)

    # using classifier18 as starting model
    classifier = SketchNet(pretrained=True)

    for idx, (name, param) in enumerate(classifier.named_parameters()):
        if idx < LAYERS_FREEZED:
            param.requires_grad = False

    classifier = classifier.to(DEVICE)

    optimizer = Adam(classifier.parameters(), lr=1e-3)

    # applying the rotation and flips because through manually checking it was observed that dataset contains flipped and rotated samples
    train_tranform = transforms.Compose(
        
        [
        transforms.Resize(299),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        ]

    )
    val_tranform = transforms.Compose(
        
        [
        transforms.Resize(224)
        ]

    )
    train_dataset = SketchDataSet(os.path.join(BASE_DIR,"dataset/train"),transform=train_tranform)
    val_dataset = SketchDataSet(os.path.join(BASE_DIR,"dataset/val"),transform=train_tranform)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=2,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,num_workers=2,shuffle=False)

    globalStats = []
    lastBestAcc = -1

    print(f"\n**** Starting Training *****\n")
    print(f"**** Train Samples: {len(train_dataset)} Validation Samples: {len(val_dataset)} *****\n")
    print(f"**** Trainable Parameters {classifier.count_trainables()} ****\n")
    epoch_ = 0
    model, optimizer, epoch_, stats = load_checkpoint("checkpoint/inception/best_acc_0.5692.pth")

    classifier.load_state_dict(model)
    globalStats = stats

    step = 0
    for epoch in range(epoch_,30):

        train_report = train(classifier,optimizer,train_loader,epoch,writer)

        val_report = validate(classifier,val_loader,epoch,writer)
        # print(val_report)

        print(f"**** train_mean_accuracy:{train_report['acc_mean']} val_mean_accuracy:{val_report['acc_mean']} *****")

        globalStats.append(
            {
                "train":train_report,
                "val":val_report
            }
        )

        currentAcc = val_report["acc_mean"]
        if currentAcc > lastBestAcc :
            print(f"**** Accuracy increased from {lastBestAcc:.4f} to {currentAcc:.4f} ****")
            save_checkpoint(classifier,epoch,optimizer,globalStats,os.path.join(CHECKPOINTS,f"best_acc_{currentAcc:.4f}.pth"))        
            print(f"**** best checkpoint saved ****")
            lastBestAcc = currentAcc

        save_checkpoint(classifier,epoch,optimizer,globalStats,os.path.join(CHECKPOINTS,f"last_acc_{currentAcc:.4f}.pth"))        
        print(f"**** checkpoint saved ****")
