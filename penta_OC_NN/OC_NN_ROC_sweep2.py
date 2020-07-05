import argparse
import numpy as np
import matplotlib.pyplot as plt
from OC_NN_model2 import Model
from sklearn import svm
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.utils as utils

from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description = 'give num')
parser.add_argument('nu_factor', type=int)
parser.add_argument('n_hidden', type=int)
args = parser.parse_args()

hidden_size_list = [32, 64, 128, 512]
hidden_size = hidden_size_list[args.n_hidden]
# nu = 0.05*args.nu_factor + 0.05   ##0.05 ~ 1
nu = 0.99
print('Model - nu : %.3f / hidden size : %d'%(nu, hidden_size))

def get_fpr_tpr(normal_pred, abnormal_pred, div_num=40):
    r_max = np.max(np.vstack((normal_pred, abnormal_pred)))
    r_min = np.min(np.vstack((normal_pred, abnormal_pred)))
    label = np.append(np.ones(len(normal_pred)), np.zeros(len(abnormal_pred)))
    
    fpr_list = []
    tpr_list = []
    for d in range(div_num):
        r = r_min + d*((r_max-r_min)/div_num)
        y_hat = np.append(np.float32(normal_pred>r), np.float32(abnormal_pred>r))
        fpr, tpr, threshold = roc_curve(label, y_hat, drop_intermediate=False)
        fpr_list.append(fpr[1])
        tpr_list.append(tpr[1])
    dauc = auc(fpr_list, tpr_list)
    return dauc, fpr_list, tpr_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'train_data9'
train_set = datasets.ImageFolder(root='dataset/preproced_data/%s/'%dataset_name,
                       transform=transforms.Compose([
#                              transforms.RandomHorizontalFlip(p=0.5),
                         transforms.Grayscale(),
                         transforms.Resize((64,64)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
train_loader = utils.data.DataLoader(dataset=train_set, batch_size=1000,shuffle=True)

for idx, (data, target) in enumerate(train_loader):
    train_x = data.cpu().numpy()

score_num = 1
dir_name = ['abnormal', 'normal']
test_set = datasets.ImageFolder(root='dataset/dongdong2/%s/'%dir_name[score_num],
                           transform=transforms.Compose([
#                              transforms.RandomHorizontalFlip(p=0.5),
                             transforms.Grayscale(),
                             transforms.Resize((64,64)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
test_loader = utils.data.DataLoader(dataset=test_set,batch_size=100,shuffle=False)

for idx, (data, target) in enumerate(test_loader):
    test_nm = data.cpu().numpy()
    
score_num = 0    
test_set = datasets.ImageFolder(root='dataset/dongdong2/%s/'%dir_name[score_num],
                           transform=transforms.Compose([
#                              transforms.RandomHorizontalFlip(p=0.5),
                             transforms.Grayscale(),
                             transforms.Resize((64,64)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
test_loader = utils.data.DataLoader(dataset=test_set,batch_size=100,shuffle=False)

for idx, (data, target) in enumerate(test_loader):
    test_ab = data.cpu().numpy()

train_x = np.reshape(train_x, (len(train_x),-1))
test_nm = np.reshape(test_nm, (len(test_nm), -1))
test_ab = np.reshape(test_ab, (len(test_ab),-1))

train_x = torch.Tensor(train_x).to(device)
test_nm = torch.Tensor(test_nm).to(device)
test_ab = torch.Tensor(test_ab).to(device)

print("data shape :", np.shape(train_x), np.shape(test_nm), np.shape(test_ab))

np.random.seed(37)
torch.manual_seed(37)
torch.cuda.manual_seed_all(37)
torch.backends.cudnn.deterministic = True

r = 1
learning_rate = 0.00005
model = nn.DataParallel(Model(train_x.size()[-1], hidden_size)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

mb_div = 1
mb_idx = int(len(train_x)/mb_div)
s = np.arange(len(train_x))

buff_auc = 0
for i in range(1001):
    np.random.shuffle(s)
    train_x = train_x[s]
    
    model.train()
    for mb in range(mb_div):
        d_train_x = train_x[mb*mb_idx:(mb+1)*mb_idx]
        
        optimizer.zero_grad()
        output, loss = model(d_train_x, r=r, nu=nu)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        
        output = output.cpu().detach().numpy()
        r = np.quantile(output, nu)
        
    if i %100==0 and i!=0:
        model.eval()
        output, _ = model(train_x, r=r, nu=nu)
        output_nm, _ = model(test_nm, r=r, nu=nu)
        output_ab, _ = model(test_ab, r=r, nu=nu)
        
        output = output.cpu().detach().numpy()
        output_nm = output_nm.cpu().detach().numpy()
        output_ab = output_ab.cpu().detach().numpy()
        
        dauc, fpr_list, tpr_list = get_fpr_tpr(output_nm, output_ab)
        if buff_auc < dauc: 
            buff_auc = dauc
            np.save('roc_npy2/nu%.3f_hid%d_fpr.npy'%(nu, hidden_size), fpr_list)
            np.save('roc_npy2/nu%.3f_hid%d_tpr.npy'%(nu, hidden_size), tpr_list)
            np.save('roc_npy2/nu%.3f_hid%d_outout_nm.npy'%(nu, hidden_size), output_nm)
            np.save('roc_npy2/nu%.3f_hid%d_output_ab.npy'%(nu, hidden_size), output_ab)
        print('[%d/1000] - AUC : %.4f'%(i,buff_auc))
        
