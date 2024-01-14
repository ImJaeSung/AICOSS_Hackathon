import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        bce_loss = nn.BCELoss(reduction = 'none')(inputs, targets)

        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

"""No validation set ver."""       
def train_full(model, optimizer, epochs, train_loader, device, loss):
    model.to(device)
    
    if loss == "focal":
        criterion = FocalLoss().to(device)
    else:
        criterion = nn.BCELoss().to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
 
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch + 1}], Train Loss : [{_train_loss:.5f}]')
        
            
        torch.save(model.state_dict(), './assets/efficient7_noval.pth')
        
    return model   

def train_E7(model, optimizer, epochs, train_loader, val_loader, device, loss):
    model.to(device)
    
    if loss == "focal":
        criterion = FocalLoss().to(device)
    else:
        criterion = nn.BCELoss().to(device)

    
    best_val_loss = float('inf') 
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
 
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch + 1}], Train Loss : [{_train_loss:.5f}], Val Loss : [{_val_loss:.5f}]')
        
            
        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            best_model = model
            torch.save(best_model.state_dict(), './assets/efficient7.pth')
        
    return best_model

def train_E4(model, optimizer, epochs, train_loader, val_loader, device, loss):
    model.to(device)
    
    if loss == "focal":
        criterion = FocalLoss().to(device)
    else:
        criterion = nn.BCELoss().to(device)

    
    best_val_loss = float('inf') 
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
 
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch + 1}], Train Loss : [{_train_loss:.5f}], Val Loss : [{_val_loss:.5f}]')
        
            
        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            best_model = model
            torch.save(best_model.state_dict(), './assets/efficient4.pth')

def validation(model, criterion, valid_loader, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for imgs, labels in tqdm(iter(valid_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            probs = model(imgs)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
    
    return _val_loss

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            probs = model(imgs)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions