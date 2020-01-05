from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import json
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import networkx as nx
from gat import GAT
            
def create_graph(data:np.ndarray, norm=True):
    zeros = np.where(data==0)
    if norm:
        norm_data = np.tril(data, -1)
        norm_data = norm_data[np.where(norm_data!=0)]
        min_ = np.min(norm_data)
        max_ = np.max(norm_data)
        data = (data - min_)/(max_ - min_)
    data[zeros] = 0
    h, w = data.shape
    assert h == w
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(h)])
    G.add_weighted_edges_from([(str(i), str(i), 1.) for i in range(h)])
    where = np.where(data != 0)
    G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], data[where]))
    return G


class KDD99(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, index):
        ins = self.instances[index]
        return ins

    def __len__(self):
        return len(self.instances)


class GATVAE(nn.Module):
    def __init__(self, nodeNum, hidden_features, latent_feature):
        super(GATVAE, self).__init__()
        self.nodeNum = nodeNum
        self.gat_in = GAT(nodeNum)

        self.encode_net = [nn.Linear(nodeNum, hidden_features[0]), nn.ReLU()]
        for i in range(len(hidden_features)-1):
            self.encode_net.extend([nn.Linear(hidden_features[i], hidden_features[i+1]), nn.ReLU()])
        self.encode_net.extend([nn.Linear(hidden_features[-1], latent_feature), nn.ReLU()])
        self.encode_net = nn.Sequential(*self.encode_net)

        self.mu = nn.Linear(latent_feature, latent_feature)
        self.sigma = nn.Linear(latent_feature, latent_feature)

        self.decode_net = [nn.Linear(latent_feature, hidden_features[-1]), nn.ReLU()]
        for i in range(len(hidden_features)-1):
            self.decode_net.extend([nn.Linear(hidden_features[len(hidden_features)-i-1], hidden_features[len(hidden_features)-i-2]), nn.ReLU()])
        self.decode_net.extend([nn.Linear(hidden_features[0], nodeNum), nn.ReLU()])
        self.decode_net = nn.Sequential(*self.decode_net)

        self.gat_out = GAT(nodeNum)

    def encode(self, x, adj):
        h = self.gat_in(x, adj)
        h = self.encode_net(h)
        return self.mu(h), self.sigma(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, adj):
        h = self.decode_net(z)
        return self.gat_out(h, adj)

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, adj)
        return x_hat, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    Mse = torch.nn.MSELoss(reduce=False, size_average=False)
    MSE = Mse(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # totalLoss = MSE + 0.00000001*KLD # fonts
    totalLoss = MSE + 0.00000001*KLD # kdd99
    # totalLoss = MSE + 0.0001*KLD # thyroid
    # totalLoss = MSE + 0.00000001*KLD # letter
    return MSE, KLD, totalLoss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        
        recon_batch, mu, logvar = model(data, adj)
        
        mse, kld, loss = loss_function(recon_batch, data, mu, logvar)
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num = len(data) * len(data[0])
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMSE: {:.6f}\tKLD: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / num, mse.sum() / num, kld.sum() / num))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch, test_loader):
    test_loss = 0

    fin_mu = []
    fin_logvar = []
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data, adj)
            mse, kld, loss = loss_function(recon_batch, data, mu, logvar)
            
            loss = loss.sum().item()
            test_loss += loss

            fin_mu.extend(mu.tolist())
            fin_logvar.extend(logvar.tolist())
    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return fin_mu, fin_logvar


def generatePara(test_loader, mode):
    
    model.eval()
    fin_mu = []
    fin_std = []
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            for instance in data:
                instance = instance.unsqueeze(dim=0)
                mu, logvar = model.encode(instance, adj)
                mu, logvar = mu.unsqueeze(dim=0), logvar.unsqueeze(dim=0)
                mu_hat = model.decode(mu, adj)
                mse, kld, loss = loss_function(mu_hat, instance, mu, logvar)
                loss = mse
                std = torch.exp(0.5*logvar)
                std = std
                fin_mu.append(loss.tolist())
                fin_std.append(std.tolist())
    np.save(datadir + mode + '_mu_loss.npy', np.array(fin_mu))
    np.save(datadir + mode + '_std.npy', np.array(fin_std))


def get_data(model):
    return list(map(lambda x: x.data, model.parameters()))


if __name__ == "__main__":
    datadir = '../data/kddcup99/'
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N', help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_instances = np.load(datadir + 'X_train.npy')
    train_labels = np.load(datadir + 'y_train.npy')

    test_anomaly_instances = np.load(datadir + 'X_test_anomaly.npy')
    test_anomaly_labels = np.load(datadir + 'y_test_anomaly.npy')

    test_normal_instances = np.load(datadir + 'X_test_normal.npy')
    test_normal_labels = np.load(datadir + 'y_test_normal.npy')

    nodeNum = train_instances.shape[1]

    graph_data = np.load(datadir + 'kdd99weight.npy')
    graph_data += np.eye(graph_data.shape[0])
    graph_data = np.nan_to_num(graph_data)
    G = create_graph(graph_data[:nodeNum, :nodeNum])
    adj = np.array(nx.normalized_laplacian_matrix(G).todense())
    adj = torch.FloatTensor(adj).to(device)
    adj.requires_grad = False

    train_dataset = KDD99(torch.FloatTensor(train_instances))
    test_anomaly_dataset = KDD99(torch.FloatTensor(test_anomaly_instances))
    test_normal_dataset = KDD99(torch.FloatTensor(test_normal_instances))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_anomaly_loader = DataLoader(test_anomaly_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('data prepared!\n')

    model = GATVAE(nodeNum, [60, 30, 10], 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch, test_anomaly_loader)
        test(epoch, test_normal_loader)
    generatePara(test_anomaly_loader, 'anomaly')
    generatePara(test_normal_loader, 'normal')

