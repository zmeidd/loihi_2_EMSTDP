from __future__ import print_function
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from skimage.transform import resize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sklearn.metrics as sk
from utils import get_measures
from utils import print_measures
from skimage.transform import resize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVED_MODEL_PATH = "loihi_cnn.pt"
DATA_PATH = "ori_imgs.npy"
LABEL_PATH = "ori_labels.npy"
#indsitribution label
INDIST_LABEL = [0,1,2,3,4]
'''
0,1,2,3,4 : in distribution
5,6,7,8,9 : out distribution
'''
class loihi_set(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.img = dataset[0]
        self.labels = dataset[1]
        self.transform = transform        
    def __getitem__(self, index):
        #resize imgs
        img_ = resize(self.img[index],(32,32,1))
        img = np.reshape(img_,(1,32,32))
        if self.transform:
            img = self.transform(img)
        return (torch.from_numpy(img).float(), self.labels[index])

    def __len__(self):
        return len(self.dataset[0])
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model,  in_loader, oe_loader, optimizer, epoch):
    model.train()
    loss_avg = 0.0
    correct = 0

    for in_set , out_set in zip(in_loader, oe_loader):
        data = torch.cat((in_set[0],out_set[0]),0)
        target = in_set[1]
        
        data = data.to(device= device)
        target = target.to(device = device)
        
        x = model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:len(in_set[0])], target)

        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        
        
         


def test(model,  test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            data = data.to(device = device)
            target = target.to(device = device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_oe(in_dis_data, in_dis_label,saved_model_path = SAVED_MODEL_PATH):
    
    print("done split data")
    
    in_dataset = loihi_set([in_dis_data,in_dis_label])
    in_loader = torch.utils.data.DataLoader(in_dataset,batch_size= 64, shuffle=True)
    
    transform_oe=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize(32)
        ])
    #====
    # OE dataset
    #====
    oe_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transform_oe)
    idx = list(range(0,20000))
    oe_set = torch.utils.data.Subset(oe_set, idx)
    oe_loader =  torch.utils.data.DataLoader(oe_set, batch_size = 64) 
    #====
    
    model = Net().to(device=device)
    optimizer = optim.Adadelta(model.parameters(), lr= 0.5)
    for epoch in range(1, 5):
        print("epoch  ",epoch)
        train(model,   in_loader,  oe_loader, optimizer, epoch)
        test(model,  in_loader)
        # scheduler.step()


    torch.save(model.state_dict(),saved_model_path)



'''
Preparing OOD data
'''
def prepare_ood_data(dsaic_chip, dsaic_label):
    in_dis_class = [0,1,2,3,4]
    in_dis_data = []
    in_dis_label = []
    out_dis_data = []
    out_dis_label = []
    for i in range(len(dsaic_chip)):
        if dsaic_label[i] in in_dis_class:
            if len(in_dis_data) == 0:
                in_dis_data = dsaic_chip[i]
            else:
                in_dis_data = np.vstack((in_dis_data,dsaic_chip[i]))
            
            in_dis_label.append(dsaic_label[i])
        else:
            if len(out_dis_data) == 0:
                out_dis_data = dsaic_chip[i]
            else:
                out_dis_data = np.vstack((out_dis_data,dsaic_chip[i]))
            out_dis_label.append(dsaic_label[i])
    return in_dis_data, in_dis_label, out_dis_data, out_dis_label





'''
OOD scores
'''
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(net, ood_loader,in_score, num_to_avg=1):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(net,ood_loader)
        print("out_score ood", np.mean(out_score))
        print("in_score ood", np.mean(in_score))
        count = 0
        for i in range(len(in_score)):
            if in_score[i]>-0.78:
                count+=1
        print("wrong ratio", count/len(in_score))
                
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)


    print_measures(auroc, aupr, fpr)


class loihi_set(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.img = dataset[0]
        self.labels = dataset[1]
        self.transform = transform        
    def __getitem__(self, index):
        #resize imgs
        img_ = resize(self.img[index],(32,32,1))
        img = np.reshape(img_,(1,32,32))
        if self.transform:
            img = self.transform(img)
        return (torch.from_numpy(img).float(), self.labels[index])

    def __len__(self):
        return len(self.dataset[0])
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output




# /////////////// Calibration Prelims ///////////////
def get_ood_scores(net, loader, in_dist=False, use_xent = False, percentile = 0.8):
    _score = []
    _right_score = []
    _wrong_score = []
    testing_num = 10000

    ood_num_examples = testing_num //2
    expected_ap = ood_num_examples / (ood_num_examples + testing_num)

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            data = data.view(-1, 1, 32, 32).cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def ood_detection(data_path = DATA_PATH,label_path =LABEL_PATH,num_trains =20000
                  ,num_ood=10000, percentile = 0.8):

    net = Net().to(device=device)
    #============ in distribution dataset==
    
    ori_data = np.load(data_path)
    ori_label =np.load(label_path)

    """
    Test 5000
    """
    idx =0
    idx_ood =0
    num_trains = num_trains
    #
    in_dis_data = np.zeros((num_trains,32,32,1))
    in_dis_label = []
    
    out_dis_data = np.zeros((num_ood,32,32,1))
    out_dis_label = []
    
    for i in range(len(ori_data)):
        img = resize(ori_data[i],(32,32,1))

        if ori_label[i] in INDIST_LABEL:
            if idx>=num_trains:
                break
            else:
                in_dis_data[idx] = img
                in_dis_label.append(ori_label[i])
                idx+=1
        elif (idx_ood< num_ood) and not(ori_label[i] in INDIST_LABEL):
            out_dis_data[idx_ood] = img
            out_dis_label.append(ori_label[i])
            idx_ood +=1
            
    
    print("labels", in_dis_label[10])
    train_oe(in_dis_data= in_dis_data, in_dis_label = in_dis_label)
    
    net.eval()
    net.load_state_dict(torch.load(SAVED_MODEL_PATH))
    in_dataset = loihi_set([in_dis_data,in_dis_label])
    in_loader = torch.utils.data.DataLoader(in_dataset,batch_size= 64, shuffle=True)
    #first test
    test(net,in_loader)
    
    '''
    get OOD score
    '''
    in_score, right_score, wrong_score = get_ood_scores(net,in_loader, in_dist=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print('Error Rate {:.2f}'.format(100*num_wrong/(num_wrong + num_right)))

    print("done in distribution testing")
    
    #=========== END in distribution testing======================
    
    
    #=========== Out distribution Testing ========================
    out_dataset = loihi_set([out_dis_data,out_dis_label])
    out_loader = torch.utils.data.DataLoader(out_dataset,batch_size= 64, shuffle=False)
    get_and_print_results(net,out_loader,in_score= in_score)
    
    #=========== End out distribution Testing ====================
    
    return np.percentile(in_score,percentile)

if __name__ == '__main__':
