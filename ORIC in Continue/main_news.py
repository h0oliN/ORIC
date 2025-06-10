import torch
import numpy as np
from models.dynamic_net import Vcnet, ORIC,TR
from data.data import get_iter
from utils.eval import curve
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
    # Calculate factual loss
    #loss_y = torch.sum(torch.square(y - y_hat))
    loss_y =0
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
def criterion(pred_outcome, gps, y, alpha=0.5, epsilon=1e-6):
    return ((pred_outcome.squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(gps + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='train with news data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/news', help='dir of data matrix')
    parser.add_argument('--data_split_dir', type=str, default='dataset/news/eval/5', help='dir data split')
    parser.add_argument('--save_dir', type=str, default='logs/news/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=2500, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    # Parameters

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3  #
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-5

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # get data
    data_matrix = torch.load(args.data_dir + '/data_matrix.pt').to(device)
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt').to(device)

    idx_train = torch.load(args.data_split_dir + '/idx_train.pt').to(device)
    idx_test = torch.load(args.data_split_dir + '/idx_test.pt').to(device)

    train_matrix = data_matrix[idx_train, :].to(device)
    test_matrix = data_matrix[idx_test, :].to(device)
    t_grid = t_grid_all[:, idx_test].to(device)

    n_data = data_matrix.shape[0]

    # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
    train_loader = get_iter(data_matrix[idx_train,:], batch_size=500, shuffle=True)
    test_loader = get_iter(data_matrix[idx_test,:], batch_size=data_matrix[idx_test,:].shape[0], shuffle=False)


    grid = []
    MSE = []

    for model_name in ['Vcnet','ORIC']:
        # import model
        if model_name == 'Vcnet' or model_name == 'ORIC':
            cfg_density = [(498, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots).to(device)
            model._initialize_weights()

        elif model_name == 'ORIC':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = ORIC(cfg_density, num_grid, cfg, degree, knots).to(device)
            model._initialize_weights()

        # use Target Regularization?
        if model_name == 'VCnet_tr' :
            isTargetReg = 1
        else:
            isTargetReg = 0

        if isTargetReg:
            tr_knots = list(np.arange(0.05, 1, 0.05))
            tr_degree = 2
            TargetReg = TR(tr_degree, tr_knots)
            TargetReg._initialize_weights()

        # best cfg for each model
        if model_name == 'Vcnet':
            init_lr = 0.0005
            alpha = 0.5
            tr_init_lr = 0.0005
            beta = 1.
        elif model_name == 'ORIC':
            init_lr = 0.0005
            alpha = 0.5
            tr_init_lr = 0.0005
            beta = 10

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

        if isTargetReg:
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        print('model = ', model_name)
        for epoch in range(num_epoch):
            if model_name == 'ORIC':
                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0].to(device)
                    x = inputs[:, 1:].to(device)
                    optimizer.zero_grad()
                    # out = model.forward(t, x)
                    pi_grid, gps, pred_outcome, z = model.forward(t, x)
                    loss_ot = objective(pi_grid, z, 0.001).to(device)
                    loss = criterion(pred_outcome, gps, y, alpha=alpha) + beta * loss_ot
                    loss.backward()
                    optimizer.step()
            else:
                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0]
                    x = inputs[:, 1:]

                    if isTargetReg:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        loss = criterion(out[1],out[0], y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                        loss.backward()
                        optimizer.step()

                        tr_optimizer.zero_grad()
                        out = model.forward(t, x)
                        z = out[2]
                        trg = TargetReg(t)
                        tr_loss = criterion_TR(out, trg, y, beta=beta)
                        tr_loss.backward()
                        tr_optimizer.step()
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion(out[1],out[0], y, alpha=alpha)
                        loss.backward()
                        optimizer.step()

            if epoch % verbose == 0:
                print('current epoch: ', epoch)
                print('loss: ', loss)

        if isTargetReg:
            t_grid_hat, mse = curve(model, test_matrix, t_grid, model_name, targetreg=TargetReg)
        else:
            t_grid_hat, mse = curve(model, test_matrix, t_grid, model_name)
        mse = float(mse)
        print('current loss: ', float(loss.data))
        print('current rmse: ', np.sqrt(mse))
        print('-----------------------------------------------------------------')

        grid.append(t_grid_hat)

    if args.plt_adrf:
        import matplotlib.pyplot as plt

        font1 = {'family': 'Arial',
                 'weight': 'normal',
                 'size': 22,
                 }

        font_legend = {'family': 'Arial',
                       'weight': 'normal',
                       'size': 22,
                       }
        plt.figure(figsize=(5, 5))

        c1 = 'gold'
        c2 = 'red'
        c3 = 'dodgerblue'

        truth_grid = t_grid[:, t_grid[0, :].argsort()]
        x = truth_grid[0, :].detach().cpu().numpy()
        y = truth_grid[1, :].detach().cpu().numpy()
        plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)

        x = grid[1][0, :].detach().cpu().numpy()
        y = grid[1][1, :].detach().cpu().numpy()
        plt.scatter(x, y, marker='h', label='ORIC', alpha=1, zorder=2, color=c2, s=20)

        x = grid[0][0, :].detach().cpu().numpy()
        y = grid[0][1, :].detach().cpu().numpy()
        plt.scatter(x, y, marker='H', label='Vcnet', alpha=1, zorder=3, color=c3, s=20)

        plt.yticks(np.arange(-2.0, 1.1, 0.5), fontsize=0, family='Arial')
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=0, family='Arial')

        plt.legend(prop=font_legend, loc='upper left')
        plt.xlabel('Treatment', font1)
        plt.ylabel('Response', font1)

        plt.savefig("News_ORIC_Vc.pdf", bbox_inches='tight')
        plt.show()
