# Learn.py - for training & saving model

import math
import os
from os import path
import random
import re
import time
import torch
import numpy as np
import pandas as pd
from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision import models
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters and configs
CFG = {
    "IMG_SIZE"      : 128,
    "EPOCHES"       : 50,
    "LEARNING_RATE" : 0.001,
    "BATCH_SIZE"    : 64,
    # "SEED"          : time.time_ns() % 2147483648,  # use time
    "SEED"          : 3141592,

    # Configurations
    "DATA_FOLDER"   : ".",
    "SAVED_MODEL"   : "./best_model.pth",
    "SAVED_MODEL_META": "./best_model_data.txt",
    "TARGET_CLASSES": 5,
}

# Fix seed
def use_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
use_seed(CFG["SEED"])

# Data loader
def load_labels(what):
    files = glob("./**/*.png", recursive=True)
    files = [[ filename, int(re.findall(r"(\d)[/\\].+.png", filename)[0]) ] for filename in files]
    files = pd.DataFrame(files, columns=["file_name", "label"])

    return files

# Dataset
class HandsignDataset(Dataset):
    def __init__(self, filepath:str, labels:list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.path = filepath
        self.labels = labels
    
    def __getitem__(self, index):
        img_path = path.join(self.path, self.labels["file_name"][index])
        image = cv2.imread(img_path)

        if self.transforms:
            image = self.transforms(image)
        
        if not self.train_mode:
            return image
        
        label = self.labels["label"][index]
        return image, label

    def __len__(self):
        return len(self.labels)

# Dataloader
def load_dataloader():
    # labels = load_labels("train")
    labels = load_labels("train").sample(frac=1.0).reset_index(drop=True)
    # filepath = path.join(CFG["DATA_FOLDER"], "train")
    filepath = ""

    # Train / Validation split
    validation_split  = int(len(labels) * 0.75)
    train_labels      = labels[:validation_split]
    validation_labels = labels[validation_split:].reset_index(drop=True)

    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),

        # transforms.ColorJitter(0.1, 0.2, 0.2, 0.3),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomPerspective(p=0.5),
        
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_vali = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load
    train_dataset = HandsignDataset(filepath, train_labels, train_mode=True, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)

    validation_dataset = HandsignDataset(filepath, validation_labels, train_mode=True, transforms=transform_vali)
    validation_loader = DataLoader(validation_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False)

    print(f"Loaded {len(train_labels)}/{len(validation_labels)} train/validation dataset")
    print(f"-> {len(train_loader)}/{len(validation_loader)} batches")
    return train_loader, validation_loader

# Model
def import_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)  # Pretrained
    model = model.to(device)
    model.classifier[1] = nn.Linear(1536, CFG["TARGET_CLASSES"], True)
    return model

# System
class LearningSystem():
    def __init__(self):
        self.model = import_model()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=CFG["LEARNING_RATE"])
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10, 0)
        self.train_dataloader, self.vali_dataloader = load_dataloader()

        self.best_accuracy = 0
        self.print_vali_table = True

    def save_model(self, epoch, accuracy):
        print(f"Saved model: #{epoch}, acc {accuracy * 100:.1f}%")

        # Write metadata
        with open(CFG["SAVED_MODEL_META"], "w") as f:
            f.write(str(epoch))
            f.write("\n")
            f.write(str(accuracy))
            f.write("\n")
        
        # Write model
        torch.save(self.model.state_dict(), CFG["SAVED_MODEL"])
    
    def load_model(self):
        epoch, accuracy = 0, 0

        self.model.load_state_dict(torch.load(CFG["SAVED_MODEL"]))
        with open(CFG["SAVED_MODEL_META"], "r") as f:
            epoch = int(f.readline())
            accuracy = float(f.readline())

        self.best_accuracy = accuracy
        print(f"Loaded model: #{epoch}, acc {accuracy * 100:.1f}%")


    def validate_batch(self, image, label):
        """받은 Batch를 검증하고 loss와 맞은 예측의 개수를 반환한다."""
        # Init
        image, label = image.to(device), label.to(device)

        # 평가, loss 계산, 예측
        logit = self.model(image)
        loss = self.criterion(logit, label)
        predictions = logit.argmax(dim=1, keepdim=True)
        correct = predictions.eq(label.view_as(predictions)).sum().item()

        if self.print_vali_table:
            for i, pred in enumerate(predictions):
                self.vali_statistics[min(label[i].item(), 10)][min(pred[0].item(), 10)] += 1
                # print(pred[0].item(), label[i].item(), (pred[0] == label[i]).item())

        # Loss, 맞은 예측의 개수 반환
        return loss, correct
    
    def validate(self, data_loader):
        self.model.eval()

        if self.print_vali_table:
            self.vali_statistics = [[0 for i in range(11)] for j in range(11)]

        # 받은 데이터로 검증
        running_loss = 0
        total_correct = 0
        with torch.no_grad():
            for img, label in tqdm(iter(data_loader), "Validating", ncols=80):
                loss, correct = self.validate_batch(img, label)
                running_loss += loss
                total_correct += correct
        
        # 출력
        loss_value = running_loss / len(data_loader)
        total = len(data_loader.dataset)
        accuracy = total_correct / total
        print(f"Loss: {loss_value:.6f}, Accuracy: {total_correct}/{total} ({accuracy*100:.1f}%)")

        if self.print_vali_table:
            print("    0   1   2   3   4   5   6   7   8   9   10")
            for label_idx, preds in enumerate(self.vali_statistics):
                print(str(label_idx).rjust(3), end=" ")
                for pred_idx, pred in enumerate(preds):
                    print(str(pred).ljust(3, (pred_idx == label_idx) and "." or " "), end=" ")
                print("")

        # 정확도
        return accuracy
    

    def train_batch(self, image, label) -> float:
        """받은 Batch를 학습하고 loss를 반환한다."""
        # Init
        image, label = image.to(device), label.to(device)
        self.optimizer.zero_grad()
        
        # 순전파, Loss 계산, 역전파
        predicted = self.model(image)
        loss = self.criterion(predicted, label)
        loss.backward()
        self.optimizer.step()

        # Loss 반환
        return loss.item()

    def train(self, data_loader):
        self.model.train()

        # 받은 데이터로 학습
        running_loss = 0.0
        for image, label in tqdm(iter(data_loader), "Training", ncols=80):
            loss = self.train_batch(image, label)
            running_loss += loss

        # 출력
        print(f"Loss: {running_loss / len(data_loader):.6f}, ")

    
    def do_epoches(self, epoches:int):
        # best_accuracy = 0

        for epoch in range(epoches):
            print("")
            print(f"=== Starting Epoch {epoch:02} ===")

            # 1. 학습
            self.train(self.train_dataloader)
            self.scheduler.step()

            # 2. 검증
            accuracy = self.validate(self.vali_dataloader)

            # 3. 평가
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_model(epoch, accuracy)
    

if __name__ == "__main__":
    system = LearningSystem()
    try:
        with open(CFG["SAVED_MODEL_META"], "r") as f:
            epoch = int(f.readline())
            accuracy = float(f.readline())
            print(f"Saved model found (#{epoch}, {accuracy * 100:.2f}%).")
            while True:
                ans = input("Load model? [Y/N]: ").lower().strip()
                if ans == "y":
                    system.load_model()
                    break
                elif ans == "n":
                    break
    except FileNotFoundError:
        pass

    system.do_epoches(CFG["EPOCHES"])
