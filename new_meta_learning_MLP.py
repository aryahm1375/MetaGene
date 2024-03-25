import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score

import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import precision_recall_fscore_support
torch.manual_seed(1)
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing
import sys
device = torch.device('cuda:0')
print(sys.argv)
global_batch = int(sys.argv[1]) #16
GCN_size = int(sys.argv[2]) #64
fine_tune_rate = 0
epoch_number = 50
alpha_rate = float(sys.argv[3])#0.2
target_data = str(sys.argv[4]) #"255"

def batch(iterable, n=1):
    arrays = []
    l = len(iterable)
    for ndx in range(0, l, n):
        arrays.append(iterable[ndx:min(ndx + n, l)])
    return arrays
def test(model, data, labels,fold, test):
    model.eval()
    labels_torch = torch.tensor(np.array(labels))
    labels = labels_torch
    data = torch.tensor(data).type(torch.float).to(device)
    with torch.no_grad():
        output = model(data).detach().cpu()
        predicted_labels = (output > 0.5).int().numpy()
        true_labels = labels.int().numpy()
        pfs = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        ACC_TS = accuracy_score(true_labels, predicted_labels)
        F = pfs[2]
        pres = pfs[0]
        rec = pfs[1]
        model.train()
        #roc_auc = roc_auc_score(true_labels, output.numpy())
        pr_auc = average_precision_score(true_labels, output.numpy())
        if test:
            with open(f'{target_data}_MLP/{fold}_{global_batch}_{GCN_size}_{str(alpha_rate).replace(".","_")}_{target_data}.txt', 'a') as the_file:
                the_file.write(f'{ACC_TS}\t{F}\t{pres}\t{rec}\t{pr_auc}\n')
        return (ACC_TS, F, pres, rec, pr_auc)

def get_batch_meta(X_sources, y_sources, X_target, y_target):
    X_Y_target = np.concatenate((X_target, np.expand_dims(y_target, axis=1)), axis=1)
    b_t =batch(X_Y_target, global_batch)
    X_target_batch = []
    y_target_batch = []
    for i in range(0, len(b_t)):
        X_target_batch.append(b_t[i][:, :-1])
        y_target_batch.append(b_t[i][:, -1])
    X_sources_batch = []
    y_sources_batch = []
    for index in range(0, len(X_sources)):
        X_val = []
        y_val = []
        x_src = X_sources[index]
        y_src = y_sources[index]
        X_Y_src = np.concatenate((x_src, np.expand_dims(y_src, axis=1)), axis=1)
        b_src = batch(X_Y_src, global_batch)
        for i in range(0, len(b_src)):
            X_val.append(b_src[i][:, :-1])
            y_val.append(b_src[i][:, -1])
        X_sources_batch.append(X_val)
        y_sources_batch.append(y_val)


    return global_batch, X_target_batch, y_target_batch, X_sources_batch,y_sources_batch,

def get_batch_finetune(X_finetune, y_finetune, batch_target):
    #batch_source1 = math.ceil(len(X_source1) / steps_per_epoch)
    #batch_source2 = math.ceil(len(X_source2) / steps_per_epoch)
    #batch_target = math.ceil(len(X_target) / steps_per_epoch)
    #print(np.shape(X_finetune), np.shape(y_finetune))
    X_Y_finetune = np.concatenate((X_finetune, np.expand_dims(y_finetune, axis=1)), axis=1)
    b_f = batch(X_Y_finetune, global_batch)
    #b_f = batch(X_Y_finetune, global_batch)
    yf_b = []
    for i in range(0, len(b_f)):
        yf_b.append(b_f[i][:, -1])
    xf_b = []
    for i in range(0, len(b_f)):
        xf_b.append(b_f[i][:, :-1])
    return xf_b, yf_b

def data_normalization(dataset1, dataset2, dataset3):
    '''print(dataset3.max(axis = 0))
    print(dataset2.max(axis=0))
    print(dataset1.max(axis=0))
    print('___________________')
    print(dataset3.min(axis=0))
    print(dataset2.min(axis=0))
    print(dataset1.min(axis=0))'''
    merges = pd.concat([dataset1.drop(['Unnamed: 0','y'],axis = 1), dataset2.drop(['Unnamed: 0','y'],axis = 1)
                           , dataset3.drop(['Unnamed: 0','y'],axis = 1)], axis = 0)
    scaler1 = preprocessing.StandardScaler()
    scaler2 = preprocessing.StandardScaler()
    scaler3 = preprocessing.StandardScaler()

    scaler1.fit(dataset1.drop(['Unnamed: 0','y'],  axis = 1))
    scaler2.fit(dataset2.drop(['Unnamed: 0', 'y'], axis=1))
    scaler3.fit(dataset3.drop(['Unnamed: 0', 'y'], axis=1))

    #scaler.fit(np.array(merges))
    return scaler1, scaler2, scaler3
    #print(merges)
    #print(len(merges))

'''class GCN2(nn.Module):
    def __init__(self, num_features, dim):
        super(GCN, self).__init__()
        self.dim = dim
        self.conv1 = SAGEConv(num_features, 1,add_self_loops=False)
        self.conv2 = GCNConv(dim, int(dim/2), add_self_loops=False)
        self.linear = nn.Linear(int(dim/2)*num_features,1)
        self.linear_modal = nn.Sequential(
            nn.Linear(695, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Tanh(),
            #nn.Linear(256, 256),
            #nn.Tanh()

            #nn.Linear(64, 8),
            #nn.ReLU(),
            #nn.Linear(8, 2),
            #nn.ReLU(),
            #nn.Linear(2, 1),
            #nn.ReLU(),

        )
        #self.y_mlp1 = nn.Linear(822, 256)
        #self.y_mlp2 = nn.Linear(256, 256)

        self.linear2 = nn.Linear(727, 1)
        #self.linea3 = nn.Linear(256, 128)
        #self.linea4 = nn.Linear(128, 1)
       # self.linea2 = nn.Linear(822, 256)

    def forward(self, x, edge_index):
        #x= torch.transpose(x, 0, 1)
        #x = torch.transpose(x, 1, 2)
        #x = torch.transpose(x, 0, 2)
        #x = x[0]
        #print(x.size())
        #y = x.squeeze(2)
        #print(x.squeeze(2).size())
        x = torch.unsqueeze(x, 2)
        y = self.linear_modal(x.squeeze(2))
        x = F.tanh(self.conv1(x, edge_index))
        #print(x)
        x = F.tanh(self.conv2(x, edge_index))
        #print('size1',x.size())

        #print('CONV1SIZE', x.size())
        #x = F.relu(self.conv1(x, edge_index))
        x = self.linear(x)
        x = F.tanh(x)

        x = torch.squeeze(x,2)
        x = torch.cat((x,y),1)
        x = self.linear2(x)

        #print('size2',x.size())

        #x = F.tanh(x)
        x = F.sigmoid(x)
        return torch.squeeze(x,1)
        #x = self.conv2(x, edge_index)
        #return F.sigmoid(x)
'''
class GCN(nn.Module):

    def __init__(self, num_features):
        super(GCN, self).__init__()
        #self.conv1 = GCNConv(1, dim, add_self_loops=True)
        #self.conv2 = GCNConv(dim, int(dim/2), add_self_loops=True)
        #self.linear = nn.Linear(int(dim/2)*num_features,1)
        self.linear_temp = nn.Linear(695, num_features)
        self.linear_temp2 = nn.Linear(num_features, int(num_features/2))
        self.linear2 = nn.Linear(int(num_features/2), 1)

    def forward(self, x):
        x = F.leaky_relu(self.linear_temp(x))
        x = F.leaky_relu(self.linear_temp2(x))
        x = self.linear2(x)
        x = F.sigmoid(x)
        return torch.squeeze(x,1)
'''class GCN(nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(1, int(num_features/2), kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(int(num_features/2), num_features, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features * 173, 1)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        return x.squeeze(1)'''

'''class RBF(nn.Module):
    def __init__(self, num_features):
        super(RBF, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_features, 695))
        self.beta = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.exp(-self.beta * torch.sum((x - self.centers) ** 2, dim=2))

class GCN(nn.Module):
    def __init__(self, num_centers):
        super(GCN, self).__init__()
        self.rbf_layer = RBF(num_centers)
        self.linear = nn.Linear(num_centers, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rbf_output = self.rbf_layer(x)
        output = self.linear(rbf_output)
        return self.sigmoid(output).squeeze(1)
'''

def fine_tune(X_f, y_f, model, X_test, y_test, edge_index, epoch_number = 2):

    inner_model = GCN( GCN_size)
    inner_model.cuda()
    inner_loss = nn.BCELoss()
    inner_model.load_state_dict(copy.copy(model.state_dict()))
    #for name, para in inner_model.named_parameters():
    #    if 'conv' in name:
    #        para.requires_grad = False
    inner_optimizer = optim.Adam(inner_model.parameters(), lr=0.0004)
    for epochf in range(0, epoch_number):
        loss_fine_epoch = []
        inner_model.train()
        for epochf in range(0,epoch_number):
            for b_index in range(0, len(X_f)):
                #print('FINE_TUNE_Y', y_f)
                #print(X_f)
                inner_optimizer.zero_grad()
                X_f_torch = torch.tensor(X_f[b_index]).type(torch.float)
                y_f_torch = torch.tensor(y_f[b_index]).type(torch.float)
                output = inner_model(X_f_torch,edge_index)
                loss_fine = inner_loss(output ,y_f_torch)
                loss_fine_epoch.append(loss_fine)
                loss_fine.backward()
                inner_optimizer.step()
        inner_model.eval()
        print('FINE_TUNE_TEST',test(inner_model,X_test, y_test, edge_index=edge_index))
        print('FINE LOSS', sum(loss_fine_epoch)/len(loss_fine_epoch))
        #print('____')
        #print('AVERAGE Source LOSS', sum(LOSS_SOURCE) / len(LOSS_SOURCE))
    #optimizer.zero_grad()
    #loss = loss_calculate(model(target), target_y)
    #loss.backward()
    #optimizer.step()
    return inner_model

def meta_loss(sources, target, optimizer,sources_labels, target_label, model, alpha, criterion):

    losses = torch.zeros(1,requires_grad=True).to(device)
    #print(len(sources))
    for i in range(0, len(sources_labels)):

        inner_model = GCN(GCN_size)
        inner_model.cuda()
        inner_loss = nn.BCELoss()
        inner_model.load_state_dict(copy.copy(model.state_dict()))
        inner_optimizer = optim.SGD(inner_model.parameters(), lr=0.0004, momentum=0.2)
        #inner_model()
        inner_optimizer.zero_grad()
        loss = inner_loss(inner_model(sources[i]), sources_labels[i])
        loss.backward()
        inner_optimizer.step()
        loss_value = inner_loss(inner_model(sources[i]), sources_labels[i])
        # print(loss_value)
        losses = losses.clone() + loss_value


    optimizer.zero_grad()
    sources_value = losses/len(losses)
    #sources_value = 0
    #sources_value = 0
    output = model(target)
    #print(losses)

    #print('SUM', sources_value)
    #print(np.shape(target))
    #print('target_label',target_label)
    loss_target = criterion(output, target_label)
    if alpha == 1:
        loss_bp = loss_target
    else:
        loss_bp = alpha* loss_target + (1-alpha)*sources_value
    #print('META_LOSS', sources_value)
    #print('TARGET_LOSS', loss_target)
    #loss_bp = loss_target
    #print('The LOSS', loss_bp)
    loss_bp.backward()
    optimizer.step()
    return loss_bp, sources_value
#target_data = "255"
'''
X_sources = []
y_sources = []

adj_matrix = np.load('common_adj_matrix.npy')
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
edge_index = adj_matrix.nonzero().t().contiguous()

dataset_255 = pd.read_csv('dataset_255.csv')
#print(dataset_255)
dataset_255 = dataset_255.sample(frac = 1).reset_index()
dataset_255 = dataset_255.drop(['index'],axis = 1)
#dataset_255 = dataset_255.sample(frac = 1)
#print(dataset_255)
y_255 = dataset_255['y'].astype(int)
#print(y_255)
#print(np.sum(y_255 ==1))
X_255 = np.array(dataset_255.drop(['Unnamed: 0','y'],axis = 1))

dataset_304 = pd.read_csv('dataset_304.csv')
dataset_304 = dataset_304.sample(frac = 1).reset_index()
dataset_304 = dataset_304.drop(['index'],axis = 1)
y_304 = dataset_304['y'].astype(int)
X_304 = np.array(dataset_304.drop(['Unnamed: 0','y'],axis = 1))

dataset_771 = pd.read_csv('dataset_771.csv')
dataset_771 = dataset_771.sample(frac = 1).reset_index()
dataset_771 = dataset_771.drop(['index'],axis = 1)

y_771 = dataset_771['y'].astype(int)
X_771 = np.array(dataset_771.drop(['Unnamed: 0','y'],axis = 1))
scaler1, scaler2, scaler3 = data_normalization(dataset_255, dataset_304, dataset_771)

X_771 = scaler1.transform(X_771)
X_255 = scaler2.transform(X_255)
X_304 = scaler3.transform(X_304)
'''
datasets = {}
X_sources = []
y_sources = []
for title in ['255','304','771','830']:
    dataset = pd.read_csv(f'dataset_{title}.csv')
    dataset = dataset.sample(frac = 1).reset_index()
    dataset = dataset.drop(['index'], axis = 1)
    datasets[title] = dataset
y_target = datasets[target_data]['y'].astype(int)
X_target = np.array(datasets[target_data].drop(['Unnamed: 0','y'],axis = 1))
scaler = preprocessing.StandardScaler()
X_target = scaler.fit_transform(X_target)
for title in ['255','304','771','830']:
    if not title == target_data:
        y = datasets[title]['y'].astype(int)
        X = np.array(datasets[title].drop(['Unnamed: 0', 'y'], axis=1))
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X_sources.append(X)
        y_sources.append(y)


kf = KFold(n_splits=10, shuffle=True, random_state=0)
criterion = nn.BCELoss()
ALL_TEST_RESULTS = []
for FOLD, (train_index, test_index) in enumerate(kf.split(X_target)):

    print(f'********************* NEW FOLD {FOLD} ********************')
    X_test = X_target[test_index]
    y_test = y_target[test_index]
    X_train_all = X_target[train_index]
    y_train_all = y_target[train_index]
    X_train = X_train_all[:int((1-fine_tune_rate)*len(X_train_all))]
    X_finetune = X_train_all[int((1 - fine_tune_rate) * len(X_train_all)):]
    y_train = y_train_all[:int((1 - fine_tune_rate) * len(y_train_all))]

    y_finetune = y_train_all[int((1 - fine_tune_rate) * len(y_train_all)):]
    batch_number, X_target_batch_train, y_target_batch_train, X_sources_batch, y_sources_batch = get_batch_meta(X_sources, y_sources, X_train, y_train)
    model = GCN(GCN_size)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    for epoch in range(0, epoch_number):
        model.train()
        print('epoch', epoch + 1)
        LOSS_VALS = []
        LOSS_SOURCE = []
        for b_index in range(0, len(X_target_batch_train)):
            #print(b_index)
            random_s = []
            for idx in range(0, len(X_sources_batch)):
                random_s.append(random.randint(0, math.ceil(len(X_sources_batch[idx])/global_batch) - 1))
            X_sources_list = []
            y_sources_list = []
            for indexs, item in enumerate(X_sources_batch):
                X_sources_list.append(torch.tensor(item[random_s[indexs]]).type(torch.float).to(device ))
            for indexs, item in enumerate(y_sources_batch):
                y_sources_list.append(torch.tensor(item[random_s[indexs]]).type(torch.float).to(device ))
            X_t_torch = torch.tensor(X_target_batch_train[b_index]).type(torch.float).to(device )
            y_t_torch = torch.tensor(y_target_batch_train[b_index]).type(torch.float).to(device )
            loss, source_loss = meta_loss(X_sources_list, X_t_torch, optimizer,y_sources_list,y_t_torch, model, alpha=alpha_rate, criterion = criterion)
            LOSS_VALS.append(loss)
            LOSS_SOURCE.append(source_loss)

        #model.eval()
        print('TEST',test(model,X_test, y_test, FOLD,True))
        print('TRAIN', test(model, X_train, y_train, FOLD,False))
        print('AVERAGE LOSS', sum(LOSS_VALS)/len(LOSS_VALS))
        print('AVERAGE Source LOSS', sum(LOSS_SOURCE) / len(LOSS_SOURCE))

        print('********* END EPOCH********')
    #model_fine_tune = fine_tune(X_f, y_f, model, X_test, y_test, edge_index)
    #ALL_TEST_RESULTS.append(test(model_fine_tune, X_test, y_test, edge_index=edge_index))
    #print(np.shape(np.array(ALL_TEST_RESULTS)))
ALL_TEST_RESULTS = np.array(ALL_TEST_RESULTS)
print('FINAL ACC:', np.mean(ALL_TEST_RESULTS[:,0]))
print('FINAL P:', np.mean(ALL_TEST_RESULTS[:, 2]))
print('FINAL R:', np.mean(ALL_TEST_RESULTS[:, 3]))
print('FINAL F:', np.mean(ALL_TEST_RESULTS[:, 1]))

            #model.eval()
#print(len(X_255))
#print(len(X_304))
#print(len(X_771))



