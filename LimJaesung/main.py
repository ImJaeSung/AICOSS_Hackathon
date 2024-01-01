#%%
import torch
import torch.optim as optim
from data_utils import *
from module import *
from model import *
from wideresnet import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from sam import SAM
from pretrained import *


import random
import os
import warnings
warnings.filterwarnings(action = 'ignore') 
# gpu 자원이 활용 가능하다면 device에 gpu를 할당
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = {
    'img_size' : 224,
    'epochs' : 20,
    'lr' : 3e-4,
    'batch_size' : 16,
    'seed' : 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config['seed']) # Seed 고정

df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

train_df, valid_df = train_valid_split(df, 0.8)
train_labels = get_labels(train_df)
valid_labels = get_labels(valid_df)

train_loader, valid_loader, test_loader = load_dataset(train_df, train_labels, valid_df, valid_labels, test_df, config['img_size'], config['batch_size'])

model = pretrained_Model()
# base_optimizer = torch.optim.SGD
# optimizer = SAM(model.parameters(), base_optimizer, lr = config['lr'], momentum = 0.9)
optimizer = optim.Adam(params = model.parameters(), lr = config['lr'])
# lr_scheduler = CosineAnnealingLR(optimizer, mode='min', factor = 0.1, patience = 3)
# infer_model = sam_train(model, optimizer, config['epochs'], train_loader, valid_loader, device)
infer_model = train(model, optimizer, config['epochs'], train_loader, valid_loader, device, loss = 'focal')

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:,1:] = preds
submit.head()
submit.to_csv('./answer8.csv', index = False)
