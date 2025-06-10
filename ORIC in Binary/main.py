import sys
import random
import torch
from config import args
from model.ORICModel import ORIC,objective
import numpy as np
import warnings
from load_data import DataLoader
warnings.filterwarnings("ignore")
print(args)

def main(idx):
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.gpu if use_cuda else "cpu")
    torch.cuda.set_device(0)
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    print('Dataset Seed: {}'.format(args.data_seed))

    Data = DataLoader(args=args, idx=idx)
    tr_loader = Data.tr_loader
    te_loader = Data.te_loader
    # in_loader=Data.in_loader
    model = ORIC(input_dim=3477,enc_h_dim=args.enc_h_dim,enc_out_dim=args.enc_out_dim)
    model.to(device)

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Error: optimizer = " + str(args.optim) + " is not defined", file=sys.stderr)
        sys.exit(1)

    # early stop parameters
    patience = 40
    best_loss = float('inf')
    early_stopping_counter = 0

    reg_alpha = args.reg_alpha
    print('Training started ...')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for ind, a, x, y in tr_loader:
            ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)
            a = a.reshape((a.shape[0], 1))
            y = y.reshape((y.shape[0], 1))
            y0, y1, t_pred,  z = model(x)
            loss = objective(x, y, a, t_pred, y0, y1,  z,alpha=args.alpha,beta=args.beta)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        epoch_loss = running_loss / len(tr_loader)
        reg_alpha = reg_alpha * args.reg_decay ** epoch
        print(f"Epoch {epoch}: loss = {epoch_loss:.4f} ")
        # evaluate PEHE on test data
        if epoch % 5 == 0:
            model.eval()
            model.to(device)

            pehe_list, amse_list = [], []
            tau_list, tau_hat_list = [], []

            with torch.no_grad():
                for _, x, y0, y1 in te_loader:
                    x = x.to(device)
                    y0 = y0.unsqueeze(1).to(device)
                    y1 = y1.unsqueeze(1).to(device)

                    y0_pred, y1_pred, _, _ = model.predict(x)

                    tau_hat = y1_pred - y0_pred
                    tau = y1 - y0

                    pehe_list.append((tau_hat - tau).pow(2))
                    amse_list.append((y1_pred - y1).pow(2) + (y0_pred - y0).pow(2))

                    tau_list.append(tau)
                    tau_hat_list.append(tau_hat)

            # Concatenate and compute metrics
            pehe = torch.cat(pehe_list).mean().sqrt().item()
            amse = torch.cat(amse_list).mean().sqrt().item()
            ate = (torch.cat(tau_list).mean() - torch.cat(tau_hat_list).mean()).abs().item()

            print(f"test_pehe: {pehe:.4f}, test_ate: {ate:.4f}, test_amse: {amse:.4f}")
        # early stop
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break
    print('Training ended')
    return pehe,ate,amse

#IHDP training
if __name__ == '__main__':
    num_of_datasets = 50
    res_sqrt_pehe = np.zeros(num_of_datasets)
    res_ate=np.zeros(num_of_datasets)
    res_sqrt_amse=np.zeros(num_of_datasets)
    # if train in IHDP-1000
    random_numbers = [116, 512, 246, 943, 748, 342, 897, 405, 216, 763, 267, 301, 451, 328, 136, 951, 499, 879, 582, 469, 259, 793, 339, 121, 594, 398, 759, 117, 923, 188, 367, 58, 849, 975, 770, 331, 638, 179, 807, 424, 32, 564, 723, 386, 489, 617, 786, 276, 532, 680, 808, 357, 752, 406, 682, 214, 910, 642, 251, 135, 171, 449, 22, 458, 779, 344, 973, 908, 759, 276, 118, 62, 316, 803, 271, 687, 850, 302, 113, 689, 635, 372, 901, 558, 540, 245, 142, 748, 68, 387, 251, 728, 998, 348, 462, 120, 601, 94, 518, 674, 839]

    for idx in range(1,num_of_datasets+1):
            res_sqrt_pehe[idx-1],res_ate[idx-1],res_sqrt_amse[idx-1] = main(idx)

    print(str(np.mean(res_sqrt_pehe)) + " +- " + str(np.std(res_sqrt_pehe)))
    print(str(np.mean(res_ate)) + " +- " + str(np.std(res_ate)))
    print(str(np.mean(res_sqrt_amse)) + " +- " + str(np.std(res_sqrt_amse)))
