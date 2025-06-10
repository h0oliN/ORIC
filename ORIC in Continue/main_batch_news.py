import torch
import math
import numpy as np
from models.dynamic_net import ORIC
from data.data import get_iter
from utils.eval import curve
import os

import argparse

import warnings
warnings.filterwarnings('ignore')
def Euc_mat(Z):
    # Calculate Euclidean distance matrix
    n = Z.shape[0]
    D = torch.zeros([n, n]) # Initialize distance matrix
    dc = torch.diag(torch.mm(Z, Z.T)).reshape(n, -1)
    dt = torch.diag(torch.mm(Z, Z.T)).reshape(-1, n)
    D = dc + dt - 2 * torch.mm(Z, Z.T)
    return D


def OT_W_torch(C, n, m, p, q, reg=0.5, iteration=20, device='cpu'):
    '''
    sinkhorn algorithm
    '''
    sigma = torch.ones(int(m), 1).float().to(device) / m
    T = torch.ones(n, m).to(device)
    C = torch.exp(-C / reg).float() + 1e-12
    p = torch.unsqueeze(p, dim=1)
    q = torch.unsqueeze(q, dim=1)
    for t in range(iteration):
        T = C * T
        for k in range(1):
            delta = p / torch.mm(T, sigma)
            sigma = q / torch.mm(torch.transpose(T, 0, 1), delta)
        T = torch.diag(delta.squeeze()) @ T @ torch.diag(sigma.squeeze())
    return T


def objective(pi_grids,  Z, reg_entropy):
    # Calculate Wasserstein loss
    D = Euc_mat(Z) # Distance matrix
    treat_num = pi_grids.shape[1]
    uni_dist = (torch.ones(Z.shape[0], dtype=torch.float32) / Z.shape[0]).to(D.device) # Uniform distribution
    loss_wass = 0
    for i in range(treat_num):
        col_i=pi_grids[:,i]
        col_i_norm=col_i/col_i.sum()
        T=OT_W_torch(D,col_i_norm.shape[0],uni_dist.shape[0],col_i_norm,uni_dist,reg_entropy,device=D.device)
        sk= torch.trace(torch.mm(D, T.T))
        loss_wass += sk
    return loss_wass


# criterion
def criterion(pred_outcome,gps, y, alpha=0.5, epsilon=1e-6):
    return ((pred_outcome.squeeze() - y.squeeze())**2).mean() - alpha * torch.log(gps + epsilon).mean()


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    parser = argparse.ArgumentParser(description='train with news data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/news', help='dir of data matrix')
    parser.add_argument('--data_split_dir', type=str, default='dataset/news/eval', help='dir of data split')
    parser.add_argument('--save_dir', type=str, default='logs/news/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=10, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    args = parser.parse_args()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Parameters

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # load
    num_dataset = args.num_dataset
    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_matrix = torch.load(args.data_dir + '/data_matrix.pt').to(device)
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt').to(device)


    cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]
    model = ORIC(cfg_density, num_grid, cfg, degree, knots).to(device)
    model._initialize_weights()
    init_lr = 0.001
    alpha = 0.1
    beta = 1.
    mse_list= []
    for _ in range(num_dataset):
        cur_save_path = save_path + '/' + str(_)
        if not os.path.exists(cur_save_path):
            os.makedirs(cur_save_path)

        idx_train = torch.load(args.data_split_dir + '/' + str(_) + '/idx_train.pt')
        idx_test = torch.load(args.data_split_dir + '/' + str(_) + '/idx_test.pt')

        train_matrix = data_matrix[idx_train, :]
        test_matrix = data_matrix[idx_test, :]
        t_grid = t_grid_all[:, idx_test]

        train_loader = get_iter(data_matrix[idx_train, :], batch_size=500, shuffle=True)
        test_loader = get_iter(data_matrix[idx_test, :], batch_size=data_matrix[idx_test, :].shape[0], shuffle=False)


        # define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd,
                                        nesterov=True)


        for epoch in range(num_epoch):
            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0].to(device)
                x = inputs[:, 1:].to(device)
                optimizer.zero_grad()
                pi_grid, gps, pred_outcome, z = model.forward(t, x)
                loss_ot = objective(pi_grid, z, 0.001).to(device)
                loss = criterion(pred_outcome, gps, y, alpha=alpha) + beta*loss_ot
                loss.backward()
                optimizer.step()

            if epoch % verbose == 0:
                print('current epoch: ', epoch)
                print('loss: ', loss)


        t_grid_hat, mse = curve(model, test_matrix, t_grid)
        mse = float(mse)
        mse_list.append(np.sqrt(mse))
        print('RMSE: ', np.sqrt(mse))
        print('-----------------------------------------------------------------')
    print("AMSE: {:.4f} +- {:.4f}".format(np.mean(mse_list), np.std(mse_list)))

