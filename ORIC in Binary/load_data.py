import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import rbf_kernel

def get_train_val_test(data, tr_ind, val_ind, te_ind):
    return data[tr_ind], data[val_ind], data[te_ind]

class DataLoader(object):
    def __init__(self, args, idx):
        self.args = args
        self._load(idx)

    def _load(self, idx):
        print('----- Loading data -----')

        """ NEWS DATA LOAD """
        X_data = pd.read_csv('data/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_'+ str(idx) + '.csv.x')
        AY_data = pd.read_csv('data/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_'+ str(idx) +'.csv.y', header=None)
        np.random.seed(self.args.data_seed)
        ## preprocess X
        X = np.zeros((5000, 3477))
        for doc_id, word, freq in zip(tqdm(X_data['5000']), X_data['3477'], X_data['0']):
            X[doc_id-1][word-1] = freq
        """ Use tfidf or not """
        if self.args.tfidf:
            transformer = TfidfTransformer()
            X_neighbor = transformer.fit_transform(X)
            X_neighbor = X_neighbor.toarray()
        else:
            X_neighbor = X
        pca = PCA(n_components=self.args.comp)
        X_neighbor = pca.fit_transform(X_neighbor)

        A = AY_data[0].values
        Y = AY_data[1].values
        Y0 = AY_data[3].values
        Y1 = AY_data[4].values
        sum1=np.sum(A)
        a=int(5000-1.5*sum1)
        ## add some noise to outcome Y
        noise1 = self.args.noise*np.random.normal(0,1,size=len(X))
        noise0 = self.args.noise*np.random.normal(0,1,size=len(X))
        Y = (Y1 + noise1) * A + (Y0 + noise0) * (1 - A)

        ind = np.arange(len(X))
        # Randomly sample 2000 samples from the treatment group for removal
        control_indices = ind[A[ind] == 0]
        to_remove_indices = np.random.choice(control_indices,a , replace=False)
         # Remove the sampled treatment indices from the training set
        ind_removed = np.setdiff1d(ind, to_remove_indices)
        # Randomly shuffle the indices before splitting
        np.random.shuffle(to_remove_indices)
        tr_ind = ind_removed[:1750]
        te_ind = ind_removed[1750:]
        sum=np.sum(A[tr_ind], axis=0)
        A_tr, A_te = A[tr_ind],  A[te_ind]
        X_tr,  X_te = X[tr_ind],  X[te_ind]
        Y_tr, Y_te = Y[tr_ind],  Y[te_ind]
        Y0_tr,Y0_te = Y0[tr_ind],  Y0[te_ind]
        Y1_tr,  Y1_te = Y1[tr_ind],Y1[te_ind]

        self.A_tr = A_tr
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.in_dim = len(X[0])
        num_samples_1 = len(np.where(A_tr == 1)[0])
        num_samples_0 = len(np.where(A_tr == 0)[0])
        self.p_tr = float(num_samples_1) / (float(num_samples_0) + float(num_samples_1))  #propensity score


        ## np.array -> torch.Tensor -> torch.utils.data.DataLoader
        tr_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(tr_ind), torch.Tensor(A_tr), torch.Tensor(X_tr), torch.Tensor(Y_tr))
        te_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(te_ind), torch.Tensor(X_te), torch.Tensor(Y0_te), torch.Tensor(Y1_te))

        self.tr_loader = torch.utils.data.DataLoader(tr_data_torch, batch_size=self.args.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.args.data_seed))

        self.te_loader = torch.utils.data.DataLoader(te_data_torch, batch_size=self.args.batch_size, shuffle=False)

        print('----- Finished loading data -----')