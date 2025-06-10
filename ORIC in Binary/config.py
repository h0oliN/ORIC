import argparse

""" Hyperparameter configuration """

parser = argparse.ArgumentParser(description='ORIC')
## training setting
parser.add_argument('--batch_size', '-b', type=int, default=16                                                                                                                                                                             ,
                    help='input mini-batch size for training')
parser.add_argument('--train_seed','-ts', type=int, default=1,
                    help='random seed for parameters and training')                    
parser.add_argument('--epochs','-e', type=int, default=7000,
                    help='number of epochs to train (default: 20)')
## cuda & gpu
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu','-gpu', type=int, default=0,
                    help='device id of gpu')
## optimization setting
parser.add_argument('--optim','-o', type=str, default='Adam',
                    help='optimization algorithm')
parser.add_argument('--lr','-lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay','-wd', type=float, default=0.005,
                    help='The strength of weight decay for Adam')          
## objective function
parser.add_argument('--imbalance_func','-imbf', type=str, default='wasserstein',
                    help='Function for balancing feature reprensetation')
parser.add_argument('--reg_alpha','-ra', type=float, default=10,
                    help='Regularization parameter for balancing feature representation')         
parser.add_argument('--reg_decay','-rd', type=float, default=0.999,
                    help='decay the strength of regularization as number of epochs increases')                      
## NN architecture parameters
parser.add_argument('--enc_h_dim','-ehd', type=int, default=300,
                    help='the feature dimension of feature representation (encoder)')
parser.add_argument('--enc_out_dim','-eod', type=int, default=50,
                    help='the output dimension of feature representation (encoder)')

## preprocess dataset
parser.add_argument('--data_seed','-ds', type=int, default=1,
                    help='random seed for dataset index')
parser.add_argument('--noise','-noise', type=float, default=0.,
                    help='The magnitude of noise (for synthetic dataset)')                    
parser.add_argument('--comp','-comp', type=int, default=4,
                    help='The size of dimension for PCA for IHDP and News datasets')
parser.add_argument('--tfidf','-tfidf', action='store_true', default=False,
                    help='Use tfidf or not for News dataset')
## hyperperparameter
parser.add_argument('--alpha', type=float, default=1,
                    help='Regularization parameter for predict t_pred')
parser.add_argument('--beta', type=float, default=10 ,
                    help='Regularization parameter for imbalance_loss')


args = parser.parse_args()
