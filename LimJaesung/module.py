import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes = 3, smoothing = 0.0, dim = -1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim = self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight = self.weight, reduction = self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
    
def sam_train(model, optimizer, epochs, train_loader, valid_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    best_val_loss = float('inf') 
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(True):
            
                output = model(imgs)
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.first_step(zero_grad = True)

                criterion(model(imgs), labels).backward()
                optimizer.second_step(zero_grad = True)

            train_loss.append(loss.item())
            
                    
        _val_loss = validation(model, criterion, valid_loader, device)
        _train_loss = np.mean(train_loss)
        
        print(f'Epoch [{epoch + 1}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}]')
            
        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            best_model = model
    
    return best_model

def train(model, optimizer, epochs, train_loader, val_loader, device, loss):
    model.to(device)
    if loss == 'smoothing':
        criterion = LabelSmoothingLoss(classes = 60, smoothing = 0.1).to(device)
    elif loss == 'focal':
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
        print(f'Epoch [{epoch + 1}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}]')
            
        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            best_model = model
    
    return best_model

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