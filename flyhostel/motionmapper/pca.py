import numpy as np
import cupy as cp
import codetiming

class PCA:
    """
    Implementation of PCA with GPU hardware acceleration
    using cupy as backend and scikit-learn like API
    """
    
    def __init__(self, n_components, whiten=False, mode=1):
        
        if whiten:
            raise NotImplementedError("whitening is not implemented")
            
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.singular_values_ = None
        self.cov_matrix = None
        self.mode=mode
        
    @staticmethod
    def pca_numpy_cupy(data):
        """
        Run a PCA using the GPU with cupy as backend
        """
        with codetiming.Timer(text="Covariance matrix took {:.8f} seconds to compute"):
            print("Computing covariance matrix")
            cov_matrix=np.cov(data)
            assert cov_matrix.shape[0] == cov_matrix.shape[1] == data.shape[0]

        with cp.cuda.Device(0):
            with codetiming.Timer(text="Sending covariance matrix to GPU in {:.8f} seconds"):
                cov_matrix_gpu = cp.asarray(cov_matrix)
            with codetiming.Timer(text="Eigendecomposition took {:.8f} seconds to compute"):
                print("Computing eigendecomposition")
                vals_unsorted_gpu, T_unsorted_gpu = cp.linalg.eigh(cov_matrix_gpu)
            with codetiming.Timer(text="Receiving eigendecomposition from GPU in {:.8f} seconds"):
                vals_unsorted = vals_unsorted_gpu.get()
                T_unsorted = T_unsorted_gpu.get()

        T=T_unsorted[:, np.argsort(vals_unsorted)[::-1]]
        vals=vals_unsorted[np.argsort(vals_unsorted)[::-1]]

        return vals, T, cov_matrix
    
    @staticmethod
    def pca_cupy(data):

       cov_matrix = cp.cov(data)
       assert cov_matrix.shape[0] == cov_matrix.shape[1] == data.shape[0]
       
       vals_unsorted, T_unsorted = cp.linalg.eigh(cov_matrix)
       
       T = T_unsorted[cp.argsort(vals_unsorted)[::-1]]
       vals = vals_unsorted[cp.argsort(vals_unsorted)[::-1]]
       
  
       
       return vals, T, cov_matrix 
        
    
    def fit(self, data):
        """
        Arguments:
            * data (np.ndarray, cp.array): D features x N observations
        Returns:
            None
        """
        if self.mode == 1:
            eigenvalues, eigenvectors, cov_matrix = self.pca_numpy_cupy(data)
        elif self.mode == 2:
            eigenvalues, eigenvectors, cov_matrix = self.pca_cupy(data)
        else:
            raise NotImplementedError()
            
        self.components_ = eigenvectors
        self.singular_values_ = eigenvalues
        self.cov_matrix = cov_matrix
        
    def transform(self, data):
        """
        Arguments:
            * data (arr like): D features x N observations
        Returns:
            * proj (arr like): D features x N observations after projection on principal components
        """
        assert self.components_ is not None
        proj=data.T.dot(self.components_)
        return proj
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)