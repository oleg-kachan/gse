#import autograd.numpy as np
import numpy as np
from scipy import linalg

from scipy.sparse import csr_matrix

#from ..base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
#from ..utils import check_array
#from ..utils.validation import check_is_fitted
import utils_local as utils

from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Rotations
from pymanopt.solvers import TrustRegions, SteepestDescent, ConjugateGradient

class GSE():
    
    def __init__(self, n_components=2, n_neighbors=5, eps=None, sigma=1.0, solver="base", max_iter=100, tol=1e-6, neighborhood_method="knn", neighbors_algorithm="auto", metric="euclidean", weighted_pca=True, weighted_ls=True, n_jobs=1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.sigma = sigma
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.neighborhood_method = neighborhood_method
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.weighted_pca = weighted_pca
        self.weighted_ls = weighted_ls
        self.n_jobs = n_jobs

        self.H = None
        self.G = None

    def _build_graph(self, X):
        """Contruction of connectivity graph G"""

        neighbors = NearestNeighbors(algorithm=self.neighbors_algorithm, metric=self.metric, n_jobs=self.n_jobs).fit(X)

        if self.neighborhood_method == "knn":
            # TODO: assert n_neighbors < number of points
            G = neighbors.kneighbors_graph(n_neighbors=self.n_neighbors, mode="distance")
        elif self.neighborhood_method == "eps_ball":
            # TODO: assert eps is not None
            G = neighbors.radius_neighbors_graph(radius=self.eps, mode="distance")
        else:
            raise ValueError("Unrecognized method of neighborhood selection='{0}'""".format(self.neighborhood_method))

        G.data = self._kernel(G.data, sigma=self.sigma)

        return G

    def _estimate_Q(self, X, G, oriented=True, weighted_pca=True):
        """Estimation of tangent space Q(X_i) at each point X_i."""

        (N, D), d = X.shape, self.n_components
        Q = np.empty((N, D, d))

        indptr = G.indptr
        indices = G.indices

        for i in range(N):
            U_idx = indices[indptr[i]:indptr[i+1]]
            U_i = X[U_idx, :] - X[i]

            if (weighted_pca==True):
                U_i = G.data[indptr[i]:indptr[i+1]].reshape(-1, 1) * U_i

            _, _, V = np.linalg.svd(U_i.T.dot(U_i))
            Q[i] = V[:d, :].T

        if oriented==True:
            Q = self._orient_Q(Q, G)

        return Q

    def _orient_Q(self, Q, G):
        """Orientation of tangent bundle Q."""
        points_order, points_close = utils.get_order(G)
        #N, p, q = Q.shape
        Q_oriented = np.empty_like(Q)
        #Q_oriented = np.nan
        #points_order, points_close = get_order(graph)
        Q_oriented[0, :, :] = Q[0, :, :]
        for point_selected, point_close in zip(points_order[1:], points_close[1:]):
            if np.linalg.slogdet(Q[point_selected, :, :].T.dot(Q_oriented[point_close, :, :]))[0] > 0:
                Q_oriented[point_selected, :, :] = Q[point_selected, :, :]
            else:
                Q_oriented[point_selected, :, 0] = Q[point_selected, :, 1]
                Q_oriented[point_selected, :, 1] = Q[point_selected, :, 0]
                Q_oriented[point_selected, :, 2:] = Q[point_selected, :, 2:]
        return Q_oriented

    def _align_H(self, Q, G):
        """Dispatch to the right submethod for alignment of tangent bundle H."""

        if self.solver == 'base':
            return self._align_H_base(Q, G)
        if self.solver == 'stiefel':
            return self._align_H_stiefel(Q, G)
        else:
            raise ValueError("Unrecognized solver='{0}'""".format(self.solver))

    def _align_H_base(self, Q, G):
        """Tangent vector field alignment via contraction mappings."""
        
        N, D, d = Q.shape

        indptr = G.indptr
        indices = G.indices

        K = G.data
        
        P = np.empty((N, D, D))
        for i in range(N):
            P[i] = Q[i].dot(Q[i].T)

        H = np.array(Q)
        I_d_diag = np.eye(D)[:, :d]

        for k in range(self.max_iter):
            for i in range(N):
                H_local = np.zeros((D, d))
                for index, K_ij in zip(indices[indptr[i]:indptr[i+1]], K[indptr[i]:indptr[i+1]]):
                    H_local[:, :] += K_ij * H[index, :, :]
                
                U, _, V = np.linalg.svd(P[i].dot(H_local))
                H[i, :, :] = U.dot(I_d_diag).dot(V)

        return H

    def _align_H_stiefel(self, Q, G):
        """Tangent vector field alignment via optimization on orthogonal group."""

        N, D, d = Q.shape

        indptr = G.indptr
        indices = G.indices

        K = G.data

        def cost(V):
            F = 0
            for i in range(N):
                for j, K_ij in zip(indices[indptr[i]:indptr[i+1]], K[indptr[i]:indptr[i+1]]):
                    f_i = K_ij * np.trace(np.dot(np.dot(V[i].T, np.dot(Q[i].T, Q[j])), V[j]))
                    F += f_i
                
            return F

        manifold = Rotations(d)
        problem = Problem(manifold=manifold, cost=cost)
        solver = SteepestDescent()

        V = solver.solve(problem, np.zeros((d, d)))

        return H

    def _embedding(self, X, weighted_ls=True):

        (N, D), d, k = X.shape, self.n_components, self.n_neighbors
        
        indptr = self.G.indptr
        indices = self.G.indices
        K = self.G.data

        # build vector x
        x = np.zeros((N*k, D))

        for i in range(X.shape[0]):
            x[i*k:(i+1)*k, :] = X[indices[indptr[i]:indptr[i+1]]] - X[i]
                
        x = x.reshape(-1, 1)

        # build matrix A
        A = np.zeros((k*N*D, d * (N-1)))

        for i in range(N):
    
            for idx, (j, K_ij) in enumerate(zip(indices[indptr[i]:indptr[i+1]], K[indptr[i]:indptr[i+1]])):
                
                K_ij = np.sqrt(K_ij)
                
                i_a, i_a_to = idx*D + (i*k*D), idx*D + (i*k*D+D)
                ji_a, ji_a_to = i*d, i*d+d
                jj_a, jj_a_to = j*d, j*d+d
            
                if (i==N-1):
                    # fill line of H_n, n==i
                    A_i = np.tile(K_ij * self.H[i], (1, N-1))
                    A[i_a:i_a_to, 0:d*(N-1)] = A_i
                    
                    # set j-th column to 2H_n
                    A[i_a:i_a_to, jj_a:jj_a_to] = K_ij * 2 * self.H[i]
                    
                elif (j==N-1):
                    # fill line of -H_n, n==j
                    A_i = np.tile(K_ij * -self.H[j], (1, N-1))
                    A[i_a:i_a_to, 0:d*(N-1)] = A_i
                    
                    # set i-th column to 2H_n
                    A[i_a:i_a_to, ji_a:ji_a_to] = K_ij * -2 * self.H[i]
                    
                else:
                    # insert -H_i for i
                    A[i_a:i_a_to, ji_a:ji_a_to] = K_ij * -self.H[i]
                    
                    # insert H_i for j
                    A[i_a:i_a_to, jj_a:jj_a_to] = K_ij * self.H[i]

        # solve linear system (w/ QR factorization)
        Q, R = np.linalg.qr(A)
        z_truncated = linalg.solve_triangular(R, Q.T.dot(x), lower=False).reshape(N-1, -1)

        # recover z_n
        z_n = -np.sum(z_truncated, axis=0)
        z = np.vstack((z_truncated, z_n))

        return z

    def _reconstruction(self, y, z):
        
        N, D, d, k = self.H.shape[0], self.H.shape[1], self.n_components, self.n_neighbors
        
        indptr = self.G.indptr
        indices = self.G.indices
        K = self.G.data

        reconstructed_data = []

        for i in range(N):
            
            kernel_sum = 0.0
            G_matrix_sum = np.zeros((D, d))
            X_sum = np.zeros(D)
            y_residuals_sum = np.zeros(d)
            
            Q_rec_point = self.Q[i]
            
            for j, K_ij in zip(indices[indptr[i]:indptr[i+1]], K[indptr[i]:indptr[i+1]]):
                
                Q_rec_neigh = self.Q[j]
                kernel_sum += K_ij
                G_matrix_sum += K_ij * self.H[j]
                X_sum += K_ij * self.X[j]
                y_residuals_sum += K_ij * (y[i] - z[j])
                
            rec_point_projector = Q_rec_point.dot(Q_rec_point.T)
            G_matrix_sum = rec_point_projector.dot(G_matrix_sum) / kernel_sum
            
            g_reconstruction = (X_sum - G_matrix_sum.dot(y_residuals_sum)) / kernel_sum
            reconstructed_data.append(g_reconstruction)
            
        reconstructed_data = np.asarray(reconstructed_data)

        return reconstructed_data

    def _kernel(self, X, L=None, sigma=1):
        return np.exp(- X ** 2 / (2 * sigma ** 2))

    def fit(self, X, y=None, embed=True):
        self.X = X
        self.G = self._build_graph(X)
        self.Q = self._estimate_Q(X, self.G, weighted_pca=self.weighted_pca)
        self.H = self._align_H(self.Q, self.G)

        return self.Q, self.H, self.G

    def transform(self, X):
        z = self._embedding(X)

        return z

    def reconstruct(self, y, z):
        X_rec = self._reconstruction(y , z)

        return X_rec