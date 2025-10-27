from __future__ import print_function
import argparse
import time
from collections import OrderedDict
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.model_selection import KFold
import random
from torch.autograd import Variable
from torch.utils.data import Dataset
from scipy import sparse
from scipy.io import loadmat
# from torch.optim.lr_scheduler import StepLR
# from torch_geometric.nn import TopKPooling
# from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torch_geometric.nn import GCNConv
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# load the data into the dataset
class GraphDataSet(Dataset):
    def __init__(self, num_data, graph_seq):
        max_node = 125
        for i in range(num_data):
            ind = graph_seq[i]
            # load files
            vornb = loadmat(r'H:\voronoi_0522\data\stru_'+str(ind)+r'\vornb.mat')
            K = loadmat(r'H:\voronoi_0522\data\stru_'+str(ind)+r'\K.mat')
            points = loadmat(r'H:\voronoi_0522\data\stru_'+str(ind)+r'\points.mat')
            vorvx = loadmat(r'H:\voronoi_0522\data\stru_'+str(ind)+r'\vorvx.mat')
            V = loadmat(r'H:\voronoi_0522\data\stru_'+str(ind)+r'\V.mat')
            
            vorvx=vorvx['vorvx']
            K=K['K1']
            vornb=vornb['vornb']
            
            file_paths = [r'H:\voronoi_0522\data\stru_{}\Material.txt'.format(ind), 
                          r'H:\voronoi_0522\data\stru_{}\label.txt'.format(ind),
                          r'H:\voronoi_0522\data\stru_{}\Euler.txt'.format(ind)]
            graph_elements = [np.loadtxt(file_paths[0]), np.loadtxt(file_paths[1]), np.loadtxt(file_paths[2])]
             
            ####125*n feature matrix
            points=points['pos']
            V=V['V']
            degree=np.zeros((125,1))
            for num in range(len(degree)):
                degree[num,0]=len(vornb[0][num][0])
            materials = graph_elements[0].reshape(125,1)
            Euler_angle = graph_elements[2]/360
            material = np.zeros((125,4))
            
            for num1 in range(len(materials)):
                if materials[num1,0] == 1:
                    material[num1,0] = 1#k
                    # material[num1,4] = 0.5757#Expansion
                    material[num1,1] = Euler_angle[num1,0]
                    material[num1,2] = Euler_angle[num1,1]
                    material[num1,3] = Euler_angle[num1,2]
                    # material[num1,1] = 0.208#Conductivity
                    
                elif materials[num1,0] == 2:
                    material[num1,0] = 0.6183#k
                    # material[num1,4] = 0.7229#Expansion
                    material[num1,1] = Euler_angle[num1,0]
                    material[num1,2] = Euler_angle[num1,1]
                    material[num1,3] = Euler_angle[num1,2]
                    # material[num1,1] = 1#Conductivity
                    
                elif materials[num1,0] == 3:
                    material[num1,0] = 0.3381#k
                    # material[num1,4] = 1#Expansion
                    material[num1,1] = Euler_angle[num1,0]
                    material[num1,2] = Euler_angle[num1,1]
                    material[num1,3] = Euler_angle[num1,2]
                    # material[num1,1] = 0.5151#Conductivity
                    
            # feature_matrix = np.hstack((points, V.T, degree, material))    
            # feature_matrix = np.hstack((V.T, degree, material))  
            feature_matrix = np.hstack((points, V.T, material))
            # manipulate_feature(feature_matrix)
            
            #1×1 mass label matrix
            label = np.array(np.absolute((graph_elements[1]-135000)/50000))

            # get the dimension of proprty
            num_properties = 1        
            #adj_matrix
            adj = np.zeros((max_node,max_node))
            for num2 in range(len(vornb[0])):
                for num3 in range(len(vornb[0][num2][0])):
                    adj[num2,vornb[0][num2][0][num3]-1] = 1
                    adj[vornb[0][num2][0][num3]-1,num2] = 1
            
            
            # index_edge = np.empty((2, 0), dtype=int)
            # for num4 in range(len(adj)):
            #     for num5 in range(num4,len(adj)):
            #         if adj[num4,num5] == 1:
            #             edge = np.array([[num4, num5], [num5, num4]])
            #             # 垂直堆叠数组（添加新的行）
            #             index_edge = np.hstack((index_edge, edge))
            # index_edge = torch.tensor(index_edge, dtype=torch.long)
            
            np.fill_diagonal(adj, 1)  # add the identity matrix主对角线填充1
            D = np.sum(adj, axis=0)  # calculate the diagnoal element of D
            D_inv = np.diag(np.power(D, -0.5))  # construct D
            adj = np.matmul(D_inv, np.matmul(adj, D_inv))  # symmetric normalization of adjacency matrix 
                     
            
            # change it to the several data points
            multiple_adj = [adj for x in range(num_properties)]
            multiple_feature = [feature_matrix for x in range(num_properties)]

                # concatenating the matrices
            if i == 0:
                adjacency_matrix, node_attr_matrix, label_matrix = multiple_adj, multiple_feature, label
            else:
                adjacency_matrix, node_attr_matrix = np.concatenate((adjacency_matrix, multiple_adj)), \
                                                                             np.concatenate((node_attr_matrix, multiple_feature))
            
                                                                                          
            if i != 0:
                label_matrix = np.vstack((label_matrix, label))#[label_matrix label]

        self.adjacency_matrix = np.array(adjacency_matrix)
        self.node_attr_matrix = np.array(node_attr_matrix)
        # self.index_edges = index_edges
        self.label_matrix = label_matrix
        print('Training Data:')
        print('adjacency matrix:\t', self.adjacency_matrix.shape)
        print('node attribute matrix:\t', self.node_attr_matrix.shape)

    def __len__(self):
        return len(self.adjacency_matrix)

    def __getitem__(self, idx):
        adjacency_matrix = self.adjacency_matrix[idx]#.todense()
        node_attr_matrix = self.node_attr_matrix[idx]#.todense()
        label_matrix = self.label_matrix[idx]

        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        node_attr_matrix = torch.from_numpy(node_attr_matrix)
        label_matrix = torch.from_numpy(label_matrix)
        return adjacency_matrix, node_attr_matrix, label_matrix

# def normalize_adj(adj):
#     np.fill_diagonal(adj, 1)  # add the identity matrix主对角线填充1
#     D = np.sum(adj, axis=0)  # calculate the diagnoal element of D
#     D_inv = np.diag(np.power(D, -0.5))  # construct D
#     adj = np.matmul(D_inv, np.matmul(adj, D_inv))  # symmetric normalization of adjacency matrix
#     # convert the feature matrix to sparse matrix
#     adj = sparse.csr_matrix(adj)
#     return adj

class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        y_mean = torch.mean(y_true)
        
        sst = torch.sum((y_true - y_mean) ** 2)
        
        sse = torch.sum((y_true - y_pred) ** 2)
        
        r2 = 1 - (sse / sst)
        
        return 1 - r2


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
        
    def forward(self, pred, target):
        error = target - pred  
        loss = torch.max(
            self.quantile * error,
            (self.quantile - 1) * error
        )
        return torch.mean(loss)

def manipulate_feature(feature):
    feature[:, [4]] = (feature[:, [4]] - np.mean(feature[:, [4]])) / np.std(
        feature[:, [4]])  # normalize 
    feature = sparse.csr_matrix(feature)
    return feature

def get_data(batch_size, idx_path, validation_index, testing_index, folds, num_data):
    indices = np.load(idx_path, allow_pickle=True)['indices']
    graph_seq = np.load(idx_path, allow_pickle=True)['graph_seq']
    validation_idx = indices[validation_index]
    test_idx = indices[testing_index]
    train_idx = indices[[i for i in range(folds) if i != validation_index and i != testing_index]]
    train_idx = [item for sublist in train_idx for item in sublist]

    dataset = GraphDataSet(num_data, graph_seq)
    train_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    validation_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_idx))
    test_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_data, validation_data, test_data

def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())

def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

def print_preds(y_label_list, y_pred_list, test_or_tr):
    length, w = np.shape(y_label_list)
    print()
    print('{} Set Predictions: '.format(test_or_tr))
    for i in range(0, length):
        print('True:{}, Predicted: {}'.format(y_label_list[i], y_pred_list[i]))

def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)

def quantile_loss(Y_prime, Y, quantile=0.5):
    error = Y - Y_prime
    return torch.mean(torch.max(quantile*error, (quantile-1)*error))

def macro_avg_err(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.sum(np.abs(Y - Y_prime)) / np.sum(np.abs(Y))
    return torch.sum(torch.abs(Y - Y_prime)) / torch.sum(torch.abs(Y))

def split_data(num_folds, num_data, random_seed):
    graph_seq = np.arange(1,num_data+1)
    random.Random(random_seed).shuffle(graph_seq)
    dataset = GraphDataSet(num_data, graph_seq)
    num_of_data = dataset.__len__()
    kf = KFold(n_splits=num_folds, shuffle=True, random_state = random_seed)
    ind = []
    for i, (_, index) in enumerate(kf.split(np.arange(num_of_data))):
        ind.append(index)
    ind = np.asarray(ind, dtype=object)
    return graph_seq, ind

def extract_graph_data(out_file_path, indices, graph_seq):
    np.savez_compressed(out_file_path, indices = indices, graph_seq = graph_seq)

class Message_Passing(nn.Module):
    def forward(self, x, adjacency_matrix):
        neighbor_nodes = torch.bmm(adjacency_matrix, x)#邻接矩阵×节点特征向量
        logging.debug('neighbor message\t', neighbor_nodes.size())
        logging.debug('x shape\t', x.size())
        return neighbor_nodes

# class Message_Passing(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.conv = GCNConv(input_dim, output_dim)
      
#     def forward(self, x, index_edges):
#         return self.conv(x, index_edges)
                         
class GraphModel(nn.Module):
    def __init__(self, max_node_num, atom_attr_dim, latent_dim1, latent_dim2):
        super(GraphModel, self).__init__()

        self.max_node_num = max_node_num
        self.atom_attr_dim = atom_attr_dim
        self.latent_dim0 = latent_dim0
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        

        self.graph_modules = nn.Sequential(OrderedDict([
            ('message_passing_0', Message_Passing()),
            ('dense_0', nn.Linear(self.atom_attr_dim, self.latent_dim1)),
            # ('dropout', nn.Dropout(0.8)),
            ('activation_0', nn.ReLU()),
            
            # ('message_passing_1', Message_Passing()),
            # ('dense_1', nn.Linear(self.latent_dim2, self.latent_dim1)),
            # ('dropout', nn.Dropout(0.7)),
            # ('activation_1', nn.ReLU()),
            
            ('message_passing_2', Message_Passing()),
            ('dense_2', nn.Linear(self.latent_dim1, self.latent_dim0)),
            # ('dropout', nn.Dropout(0.6)),
            ('activation_2', nn.ReLU())
            
        ]))
        
        # self.graph_modules = nn.Sequential(OrderedDict([
        #     ('message_passing_0', Message_Passing()),
        #     ('dense_0', nn.Linear(self.atom_attr_dim, self.latent_dim1)),
        #     ('dropout', nn.Dropout(0.5)),
        #     ('activation_0', nn.ReLU()),
        #     ('message_passing_1', Message_Passing()),
        #     ('dense_1', nn.Linear(self.latent_dim1, self.latent_dim0)),
        #     ('dropout', nn.Dropout(0.3)),
        #     ('activation_1', nn.ReLU()),
        # ]))

        self.fully_connected = nn.Sequential(
            nn.Linear(self.max_node_num * self.latent_dim0, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
           
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(), 
            
            nn.Linear(128, 1)
        )

        return

    def forward(self, node_attr_matrix, adjacency_matrix):
        node_attr_matrix = node_attr_matrix.float()
        adjacency_matrix = adjacency_matrix.float()
        x = node_attr_matrix
        logging.debug('shape\t', x.size())

        for (name, module) in self.graph_modules.named_children():
            if 'message_passing' in name:
                x = module(x, adjacency_matrix=adjacency_matrix)
            else:
                x = module(x)

        # Before flatten, the size should be [Batch size, max_node_num, latent_dim]
        logging.debug('size of x after GNN\t', x.size())
        # After flatten is the graph representation
        x = x.view(x.size()[0], -1)
        logging.debug('size of x after GNN\t', x.size())

        # Concatenate [x, t]
        # x = torch.cat((x, t_matrix), 1)#将x与t_matrix拼接在一起
        x = self.fully_connected(x)
        return x

def r2_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps=1e-8) -> torch.Tensor:
    # y_pred_flat = y_pred.view(-1)
    # y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred
    y_true_flat = y_true
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2)
    y_mean = torch.mean(y_true_flat)
    ss_tot = torch.sum((y_true_flat - y_mean) ** 2)
    r2 = 1 - (ss_res / (ss_tot + eps))
    return 1 - r2  

def train(model, train_data_loader, validation_data_loader, epochs, checkpoint_dir, optimizer, criterion, validation_index, folder_name):
    print()
    print("*** Training started! ***")
    print()
    
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.5)#变
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    filename='{}/learning_Output_{}.txt'.format(folder_name, validation_index)
    output=open(filename, "w")
    print('Epoch Training_time Training_MSE Validation_MSE R2',file=output, flush = True)  
    for epoch in range(epochs):
        model.train()
        total_macro_loss = []
        total_mse_loss = []
      #  if epoch % (epochs / 10) == 0 or epoch == epochs-1:
      #      torch.save(model.state_dict(), '{}/checkpoint_{}.pth'.format(checkpoint_dir, epoch))
      #      print('Epoch: {}, Checkpoint saved!'.format(epoch))
      #  else:
      #      print('Epoch: {}'.format(epoch))

        train_start_time = time.time()
        
        for batch_id, (adjacency_matrix, node_attr_matrix, label_matrix) in enumerate(train_data_loader):
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)
            label_matrix = tensor_to_variable(label_matrix)

            optimizer.zero_grad()

            y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix)
            loss = criterion(y_pred, label_matrix)
            total_macro_loss.append(macro_avg_err(y_pred, label_matrix).item())
            total_mse_loss.append((loss.item()))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        train_end_time = time.time()
        _, training_loss_epoch = test(model, train_data_loader, 'Training', False, criterion, validation_index, folder_name) 
        _, validation_loss_epoch = test(model, validation_dataloader, 'Validation', False, criterion, validation_index, folder_name)
        print('%d %.3f %e %e' % (epoch, train_end_time-train_start_time, training_loss_epoch, validation_loss_epoch), file=output, flush=True )
    return total_macro_loss

def test(model, data_loader, test_val_tr, printcond, criterion, running_index, folder_name):
    model.eval()
    if data_loader is None:
        return None, None

    y_label_list, y_pred_list, total_loss = [], [], 0

    for batch_id, (adjacency_matrix, node_attr_matrix, label_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)
        node_attr_matrix = tensor_to_variable(node_attr_matrix)
        label_matrix = tensor_to_variable(label_matrix)

        y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix)
        
        y_label_list.extend(variable_to_numpy(label_matrix))
        y_pred_list.extend(variable_to_numpy(y_pred))

    # norm = np.load('norm.npz', allow_pickle=True)['norm']
    # label_mean, label_std = norm[0], norm[1]

    y_label_list = np.array(y_label_list) * 50000 + 135000
    y_pred_list = np.array(y_pred_list) * 50000 + 135000
    # r2 = r2_score(y_label_list, y_pred_list)
    
    
    total_loss = r2_loss(torch.from_numpy(y_pred_list), torch.from_numpy(y_label_list))
    # total_loss = macro_avg_err(y_pred_list, y_label_list)
    total_mse = criterion(torch.from_numpy(y_pred_list), torch.from_numpy(y_label_list)).item()

    length, w = np.shape(y_label_list)
    if printcond:
        filename = '{}/{}_Output_{}.txt'.format(folder_name, test_val_tr, running_index)
        output = open(filename, 'w')
        #print()
        print('{} Set Predictions: '.format(test_val_tr), file = output, flush = True)
        print('True_value Predicted_value', file=output, flush = True)
        for i in range(0, length):
            # print('%f, %f' % (y_label_list[i], y_pred_list[i]),file=output,flush = True)
            print('%f, %f' % (float(y_label_list[i]), float(y_pred_list[i])), file=output, flush=True)

    return total_loss, total_mse


######start=====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=8)
    parser.add_argument('--num_data', type=int, default=2000)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--max_node_num', type=int, default=125)
    parser.add_argument('--atom_attr_dim', type=int, default=8)
    parser.add_argument('--num_graphs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--min_learning_rate', type=float, default=0)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/')
    parser.add_argument('--validation_index', type=int, default=0)
    parser.add_argument('--testing_index', type=int, default=1)
    parser.add_argument('--idx_path', type=str, default='indices_and_graphseq.npz')
    parser.add_argument('--folder_name', type=str, default='output/')
    
    losses = []
    accuracies = []
    given_args = parser.parse_args()
    num_folds = given_args.folds
    num_data = given_args.num_data
    random_seed = given_args.random_seed
    out_file_path = 'indices_and_graphseq.npz'    
    max_node_num = given_args.max_node_num
    atom_attr_dim = given_args.atom_attr_dim
    num_graphs = given_args.num_graphs
    checkpoint_dir = given_args.checkpoint
    validation_index = given_args.validation_index
    testing_index = given_args.testing_index
    idx_path = given_args.idx_path
    folds = given_args.folds
    batch_size = given_args.batch_size
    min_learning_rate = given_args.min_learning_rate
    seed = given_args.seed
    folds = given_args.folds
    folder_name = given_args.folder_name
    
    #spilt the data
    print("Output File Path: {}".format(out_file_path))
    graph_seq, indices = split_data(num_folds, num_data, random_seed)
    extract_graph_data(out_file_path, indices = indices, graph_seq = graph_seq)
    print("Data successfully split into {} folds!".format(num_folds))
     
    #环境，随机数种子设置
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)   
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # os.environ['PYTHONHASHargs.seed'] = str(given_args.seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"#优化gpu工作配置
    
    os.environ["PYTHONHASHSEED"] = str(given_args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    np.random.seed(given_args.seed)
    torch.manual_seed(given_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(given_args.seed)
        torch.cuda.manual_seed_all(given_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
    latent_dim0=32
    latent_dim1=128
    latent_dim2=256
    epochs=200
    learning_rate=0.001
    
    # Define the model
    model = GraphModel(max_node_num, atom_attr_dim, latent_dim1, latent_dim2)
    if torch.cuda.is_available():
        model.cuda()
        
    # optimizer = optim.SGD(model.parameters(),lr=learning_rate)  
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)  
      
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = QuantileLoss(quantile=0.5) 
    
    
    
    # get the data
    train_dataloader, validation_dataloader, test_dataloader = get_data(batch_size, idx_path, validation_index, testing_index, folds, num_data)
    
    # train the model
    train_start_time = time.time()
    train(model, train_dataloader, validation_dataloader,epochs, checkpoint_dir, optimizer, criterion, validation_index,folder_name)
    train_end_time = time.time()
    
    torch.save(model, '{}/checkpoint.pth'.format(checkpoint_dir))    
    
    # predictions on the entire training and test datasets
    train_rel, train_mse= test(model, train_dataloader, 'Training', True, criterion, validation_index, folder_name)
    validation_rel, validation_mse=test(model, validation_dataloader, 'Validation', True, criterion, validation_index, folder_name)
    test_rel, test_mse= test(model, test_dataloader, 'Test', True, criterion, testing_index, folder_name)
    
    print('-------output---------')
    print("training_time : {}".format(train_end_time-train_start_time))
    # print("Train Relative Error: {:.3f}%".format(100*train_rel))
    # print("Test Relative Error: {:.3f}%".format(100*test_rel))
    print("Train R2: {:.3f}".format(1-train_rel))
    print("Validation R2: {:.3f}".format(1-validation_rel))
    print("Test R2: {:.3f}".format(1-test_rel))