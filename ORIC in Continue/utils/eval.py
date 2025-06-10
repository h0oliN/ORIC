import torch
import numpy as np
import json
from data.data import get_iter

def curve(model, test_matrix, t_grid,model_name, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test).to(test_matrix.device)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
    if model_name =='ORIC':
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            pi_grid,gps, pred_outcome,z = model.forward(t, x)
            out = pred_outcome.data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        if targetreg is None:
            for _ in range(n_test):
                for idx, (inputs, y) in enumerate(test_loader):
                    t = inputs[:, 0]
                    t *= 0
                    t += t_grid[0, _]
                    x = inputs[:, 1:]
                    break
                out = model.forward(t, x)
                out = out[1].data.squeeze()
                out = out.mean()
                t_grid_hat[1, _] = out
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
            return t_grid_hat, mse
        else:
            for _ in range(n_test):
                for idx, (inputs, y) in enumerate(test_loader):
                    t = inputs[:, 0]
                    t *= 0
                    t += t_grid[0, _]
                    x = inputs[:, 1:]
                    break
                out = model.forward(t, x)
                tr_out = targetreg(t).data
                g = out[0].data.squeeze()
                out = out[1].data.squeeze() + tr_out / (g + 1e-6)
                out = out.mean()
                t_grid_hat[1, _] = out
            mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
            return t_grid_hat, mse
