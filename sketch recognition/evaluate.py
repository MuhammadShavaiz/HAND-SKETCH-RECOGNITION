"""
RUN THIS FILE TO EVALUTE PERFORMANCE ON ANY DATASET

use command `python3/python evaluate.py pathToDataset`

pathToDataset must have following strucutre

    -> images
        -> 1.png
        -> 2.png
        ........
    -> labels.csv

the output is generated in the form of csv in current directory as evaluation.csv
"""
import sys
import torch
from model import SketchNet
from dataset import SketchDataSet
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


DEVICE = "cuda" if torch.cuda.is_available()  else "cpu"
CHECKPOINT_PATH = "checkpoint/best_acc_0.1855.pth"




def load_checkpoint(save_path):
    # load model weights from model_path
    saved_dict = torch.load(save_path)
    model = saved_dict["model"]
    optimizer = saved_dict["optimizer"]
    epoch = saved_dict["epoch"]

    print("**** model loaded from " + save_path + "****")

    return model, optimizer, epoch

def calculate_metrics(logits,target):
    
    """compute classification accuracy"""    


    indicies = torch.argmax(logits,dim=1)
    correctClassification = torch.sum(torch.argmax(logits,dim=1) == target, dim=0)

    accuracy = correctClassification / logits.shape[0]


    return indicies, accuracy.item()

def test(model, data_loader):

    model.eval()

    acc_train = []

    with torch.no_grad():

        with tqdm(data_loader, unit="batch", leave=False) as tepoch:
            
            for labels,images, fileNames in tepoch:
                labels = labels.to(DEVICE)
                images = images.to(DEVICE)
            
                logits = model(images)
                indicies,accuracy = calculate_metrics(logits,labels)


                indicies = torch.unsqueeze(indicies,dim=-1)
                fileNames = torch.unsqueeze(fileNames,dim=-1)

                results = torch.cat((fileNames,indicies),dim=-1).tolist()

                acc_train.append(accuracy)


                tepoch.set_postfix(
                    accuracy=100.0 * accuracy
                )


                with open("./evaluation.csv",'a') as file:
                    for res in results:
                        file.write(f"{res[0]},{res[1]}\n")


                               
            
            
                mean_acc = np.array(acc_train).mean(),
            



            return mean_acc


if __name__ == "__main__":
    args = sys.argv[1:]
    assert len(args) == 1, "use command `python3 evaluate.py pathToDataset`"
    PATH = args[0]
    dataset = SketchDataSet(PATH,testMode=True)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=False)
    #loading model
    classifier = SketchNet(pretrained=True)
    modelSate,_,_ = load_checkpoint(CHECKPOINT_PATH)
    classifier.load_state_dict(modelSate)
    acc = test(classifier,dataloader)
    print(f"**** accuracy: {acc} ****")


