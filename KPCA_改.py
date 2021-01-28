import torch
from scipy.spatial.distance import pdist, squareform
import numpy as np


class kPCA:
    def __init__(self, n_components, batch_size=None, kernel='rbf'):  # 8, 7
        self.n_components = n_components
        self.batch_size = batch_size
        self.cov = None
        self.kernel = kernel

    def partial_fit(self, X):  # X.shape  [batch_size, 49, 484]
        print('KPCA')
        if self.cov is None:
                self.cov = torch.zeros(X.shape[1], X.shape[1]).float()  # cov 49*49
        if self.kernel == 'None':
            # cov_sum.shape [49, 49]
            cov_sum = torch.matmul(X, X.transpose(-1, -2)).sum(dim=0)  # .matmul()是5000张图像与各自的转置矩阵乘积在求和
            cov_sum /= X.shape[0] * X.shape[2]
        else:
            if self.kernel == 'log':
                dists = pdist(X[0]) ** 0.2
                mat = squareform(dists)
                K = -np.log(1 + mat)
                X_1 = torch.tensor(K)
                for img in range(1, X.shape[0]):
                    dists = pdist(X[img]) ** 0.2
                    mat = squareform(dists)
                    K = torch.tensor(-np.log(1 + mat))
                    X_1 = torch.cat((X_1, K), 0)
                cov_sum = torch.matmul(X_1, X_1.transpose(-1, -2)).sum(dim=0)  # .matmul()是5000张图像与各自的转置矩阵乘积在求和
                cov_sum /= X_1.shape[0] * X_1.shape[2]
            elif self.kernel == 'rbf':
                print(X.shape[0])
                dists = pdist(X[0].t()) ** 2
                mat = squareform(dists)
                beta = 10
                K = np.exp(-beta * mat)
                X_1 = torch.tensor(K)
                X_1 = X_1[None, ...]
                for img in range(1, X.shape[0]):
                    dists = pdist(X[img].t()) ** 2
                    mat = squareform(dists)
                    beta = 10
                    K = torch.tensor(np.exp(-beta * mat))
                    K = K[None, ...]
                    X_1 = torch.cat((X_1, K), 0)
                cov_sum = torch.matmul(X_1, X_1.transpose(-1, -2)).sum(dim=0)  # .matmul()是5000张图像与各自的转置矩阵乘积在求和
                cov_sum /= X_1.shape[0] * X_1.shape[2]
        self.cov += cov_sum
        (e, v) = torch.eig(self.cov, eigenvectors=True)
        _, indicies = torch.sort(e[:, 0], descending=True)
        print('KPCA_END')
        V = X.mm(v[:, indicies[:self.n_components]]).mean(dim = 0)
        return V

'''            
    def components_(self):
       (e, v) = torch.eig(self.cov, eigenvectors=True)
       _, indicies = torch.sort(e[:, 0], descending=True)
       return torch.tensor(v[:, indicies[:self.n_components]])        # v.shape torch.Size([49, 8])
    def fit(self, X):
        print('KPCA')
       # n_samples = X.shape[0]        # torch.Size([44, 343, 202675])

       # if self.batch_size is None:
       #     batch_size_ = n_samples
       # else:
       #     batch_size_ = self.batch_size
        if self.cov is None:
            self.cov = torch.zeros(X.shape[1], X.shape[1]).float()  # cov 49*49

       # for batch in gen_batches(n_samples, batch_size_,
        #                         min_batch_size=self.n_components or 0):  # gen_batches 对n_samples进行切片处理步长是batch_size_
        cov_1 = self.partial_fit(X)
        #cov_1 = np.array(cov_1)
        #cov_1 = cov_1.astype(float)
        #cov_1 = torch.tensor(cov_1)
        print(type(cov_1), type(self.cov))
        #cov_sum = torch.tensor(cov_sum).float()
        self.cov += cov_1

        return self

   # @property
   
   
   
    def components_(self):
      (e, v) = torch.eig(self.cov, eigenvectors=True)
      _, indicies = torch.sort(e[:, 0], descending=True)
      return v[:, indicies[:self.n_components]]        # v.shape torch.Size([49, 8])
  
     '''
