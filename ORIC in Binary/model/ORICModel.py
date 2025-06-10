
import torch
import torch.nn as nn
import torch.nn.functional as F
# =============================================================================
# Adapted from:
#     "DragonNet:Adapting Neural Networks for the Estimation of Treatment Effects" implementation
#     Original author: Jonathan Schwab et al.
#     Source: https://github.com/claudiashi57/dragonnet
#     License: Apache-2.0
#
# Modifications:
#     - replace regularization terms and use OT to balance t_pred to reduce selection bias
#     -
# =============================================================================

class ORIC(nn.Module):
    """
    Base Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    enc_h_dim: int
        layer size for hidden shared representation layers
    enc_out_dim: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, enc_h_dim=80, enc_out_dim=50):
        super(ORIC, self).__init__()

        self.fc1 = nn.Linear(input_dim, enc_h_dim)
        self.fc2 = nn.Linear(enc_h_dim, enc_h_dim)
        self.fcz = nn.Linear(enc_h_dim, enc_h_dim)


        self.treat_out = nn.Linear(enc_h_dim, 1)

        self.y0_fc1 = nn.Linear(enc_h_dim, enc_out_dim)
        self.y0_fc2 = nn.Linear(enc_out_dim, enc_out_dim)
        self.y0_out = nn.Linear(enc_out_dim, 1)

        self.y1_fc1 = nn.Linear(enc_h_dim, enc_out_dim)
        self.y1_fc2 = nn.Linear(enc_out_dim, enc_out_dim)
        self.y1_out = nn.Linear(enc_out_dim, 1)


    def forward(self, inputs):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fcz(x))

        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = F.relu(self.y0_fc1(z))
        y0 = F.relu(self.y0_fc2(y0))
        y0 = self.y0_out(y0)

        y1 = F.relu(self.y1_fc1(z))
        y1 = F.relu(self.y1_fc2(y1))
        y1 = self.y1_out(y1)

        return y0, y1, t_pred, z

    def predict(self, x):
        """
        Function used to predict on covariates.

        Parameters
        ----------
        x: torch.Tensor or numpy.array
            covariates

        Returns
        -------
        y0_pred: torch.Tensor
            outcome under control
        y1_pred: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        """
        x = torch.Tensor(x)
        with torch.no_grad():
                y0_pred, y1_pred, t_pred,z = self.forward(x)
        return y0_pred, y1_pred, t_pred, z


def objective(x,y_true, t_true, t_pred, y0_pred, y1_pred,z, alpha=1, beta=1):
    """
    Generic loss function for ORIC

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted potential variable under control
    y1_pred: torch.Tensor
        Predicted potential variable under treatment
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    beta: float
        loss component weighting hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    t_pred = (t_pred + 0.01) / 1.02
    loss_t = torch.sum(F.binary_cross_entropy(t_pred, t_true))

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_y = loss0 + loss1

    imbalance_loss = wasserstein(x, z, t_pred)

    return loss_y+alpha*loss_t+beta*imbalance_loss




# calculate wasserstein distance
def wasserstein(x,x_enc, t_pred,its=10,lamb=10, ite=10,  backpropT=False):
    # compute distance matrix
    epsilon = 1e-9
    device = x_enc.device
    Mx_10 = torch_pdist2(x_enc, x_enc).to(device)
    # estimate lambda & delta
    Mx_10_mean = torch.mean(Mx_10)
    # torch_dropout = torch.nn.dropout(10 / (num_samples_0 * num_samples_1))
    # Mx_10_drop = torch_dropout(Mx_10)
    delta = torch.max(Mx_10).detach()  # detach() = no gradient computed
    eff_lamb = (lamb / Mx_10_mean).detach()
    # compute new distance matrix
    Mx_10_new = Mx_10.to(device)
    row = delta * torch.ones(Mx_10[0:1, :].shape).to(device)
    col = torch.cat([delta * torch.ones(Mx_10[:, 0:1].shape).to(device), torch.zeros((1, 1)).to(device)], 0)
    Mx_10_new = torch.cat([Mx_10, row], 0)
    Mx_10_new = torch.cat([Mx_10_new, col], 1)
    # compute kernel matrix
    Mx_10_lamb = eff_lamb * Mx_10_new
    Kx_10 = torch.exp(- Mx_10_lamb) + 1.0e-06  ## constant added to avoid nan
    def compute_wasserstein_distance(marginal1, marginal0):
        U = Kx_10 * Mx_10_new
        marginal1invK = Kx_10 / (marginal1 + 1e-8)
        # Fixed-point iterations of Sinkhorn algorithm
        u = marginal1 + epsilon
        for i in range(ite):
            inner_term = torch.matmul(torch.t(u), Kx_10)
            inner_term = torch.clamp(inner_term, min=epsilon, max=1e+8)
            inv_inner_term = 1.0 / inner_term
            u = 1.0 / (torch.matmul(marginal1invK, (marginal0 / (inv_inner_term + epsilon)))) + epsilon
            u = torch.clamp(u, min=epsilon, max=1e+8)
        v = marginal0 / (torch.t(torch.matmul(torch.t(u), Kx_10)))
        T = u * (torch.t(v) * Kx_10)
        if backpropT is False:
            T = T.detach()
        E = T * Mx_10_new
        D = 2 * torch.sum(E)
        return D
    n = len(t_pred)
    uniform_treatment_scores = torch.full((n,), 1 / n, dtype=torch.float32).to(x.device)
    total_sum1 = torch.sum(t_pred)
    normalized_propensity_scores = t_pred / total_sum1
    normalized_propensity_scores = torch.cat(
        [normalized_propensity_scores, torch.zeros((1, 1)).to(device)], 0)
    t1=1-t_pred
    total_sum0 = torch.sum(t1)
    normalized_non_treatment_scores=t1/total_sum0
    normalized_non_treatment_scores = torch.cat(
        [normalized_non_treatment_scores, torch.zeros((1, 1)).to(device)], 0)
    uniform_treatment_scores = torch.cat([uniform_treatment_scores.unsqueeze(1), torch.zeros((1, 1)).to(device)], 0)
    D1 = compute_wasserstein_distance(normalized_propensity_scores, uniform_treatment_scores)
    D0 = compute_wasserstein_distance(normalized_non_treatment_scores, uniform_treatment_scores)
    return D1 + D0 # , Mx_10_lamb

# calculate Euclidean distance
def torch_pdist2(X, Y):
    nx = torch.sum(torch.square(X), dim=1, keepdim=True)
    ny = torch.sum(torch.square(Y), dim=1, keepdim=True)
    C = -2 * torch.matmul(X, torch.t(Y))
    D = (C + torch.t(ny)) + nx
    D=torch.clamp(D, min=1e-8)
    return D

