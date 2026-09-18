from gudhi.point_cloud.dtm import DistanceToMeasure
from scipy.sparse.csgraph import shortest_path as SP
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import networkx as nx

import numpy as np
from numpy.linalg import norm

import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from time import time

# Typing
from typing import Callable
from numpy import ndarray
from dataclasses import dataclass


def pairwise_dist(X, **kwargs):
    '''
    Computes Euclidean distances between all pairs of points in `X`.
    '''
    return cdist(X, X)

def Euclidean_dist(X, i, j):
    '''
    Computes Euclidean distances between pairs (i, j) of points in `X`.
    `i` and `j` are indices or same-size arrays of indices for points in `X`.
    If single indices, the result will be an array of shape (1,).
    '''
    i, j = np.atleast_1d(i), np.atleast_1d(j)  # Ensure i and j are arrays
    return norm(X[i] - X[j], axis=1)


class Metric:
    '''
    Metric on a graph of vertices `X`.
    `select_edges` method is optionnaly used to restrict the set of edges.
    `f_edge` method returns the distances between all pairs of points. If it allows for an argument `A`, it ignores unwanted edges. It must be defined in order to use `make_edges`.
    `make_edges` method constructs edges using `f_edge`. 
    `shortest_path` method recovers all distances when the metric is non-euclidean.
    '''
    def __init__(self, X:ndarray[float, (...,...)]):
        self.X = X
        self.G = nx.Graph()
        for i, x in enumerate(X):
            self.G.add_node(i, pos=x)
        self.N = len(X)  # Size of the graph
        
        self.edges = None  # Treated as all edges admissible
        self.N_edges = self.N * (self.N-1) // 2  # Number of edges in a complete graph

        # By default, distances and paths are dictionnaries: dic[s][t] contains the distance or path from s to t.
        # If `shortest_path` is called for all sources, distances will be overwritten with a (N, N) array of distances and paths need to be computed individually from the predecessor matrix. 
        self.distances = {}
        self.paths = {}
        return

    def select_edges(self, pairs:ndarray[int, (...,2)]=None, knn:int=None, epsilon:float=None, **kwargs):
        '''
        Selects the edges to appear in the graph. The edges are not yet stored in the nx graph.
        Defines `self.edges` an array of indices of shape (N_edge, 2) or None if all edges are admissible.
        If `pairs` is given, then simply use it as the list of edges. Else use either `knn` or `epsilon` method to select edges.
        '''
        if pairs is not None:
            self.edges = np.atleast_2d(pairs)
        elif epsilon is not None:
            assert knn is None, "epsilon and knn cannot be simultaneously used."
            tree = cKDTree(self.X)  # Use a KDTree for efficient edge selection
            if epsilon is not None:
                self.edges = np.array(list(tree.query_pairs(epsilon)))  # Convert to numpy array of indices of shape (N_pairs, 2)
        elif knn is not None:
            tree = cKDTree(self.X)  # Use a KDTree for efficient edge selection
            knn = min(knn, self.N-1)  # Ensure knn is at most G.N-1
            _, indices = tree.query(self.X, k=knn+1)
            # Get all knn pairs (exclude self-loops), remove duplicates by sorting and using a set
            edge_set = set()
            for i in range(self.N):
                for j in range(1, knn+1):
                    edge_set.add(tuple(sorted((i, indices[i, j]))))
            self.edges = np.array(list(edge_set))
        return self
        
    def make_edges(self, **kwargs):
        '''
        Computes the edge weights using a method `f_edge` or `f_edge_all` that need to be defined before, and adds them to the nx graph.
        `self.f_edge` needs to support two arguments as indices or arrays of indices, and computes the edge weights for these endpoints.
        `self.f_edge_all` requires no argument and computes all edges.
        Possible kwargs are passed to `f_edge` or `f_edge_all`.
        '''
        if self.edges is None:  # Complete graph
            assert hasattr(self, 'f_edge_all'), "`f_edge_all` was not defined."
            weights = self.f_edge_all(**kwargs)
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    self.G.add_edge(i, j, weight=weights[i, j])
        elif len(self.edges) > 0:  # Compute weights only for selected edges
            assert hasattr(self, 'f_edge'), "`f_edge` was not defined."
            weights = self.f_edge(self.edges[:, 0], self.edges[:, 1], **kwargs)
            for (i, j), w in zip(self.edges, weights):
                self.G.add_edge(i, j, weight=w)
        return self
    
    def shortest_path(self, source:int|ndarray[int, (...,)]=None):
        '''
        Computes the shortest paths to get the graph distances from `source`.
        `source` can be an indice or an array of indices. If not provided then compute all paths.
        In the first case, self.distances_array and self.paths are dictionnaries containing distance and paths.
        In the second case, self.distances_array and self.predecessors are (N, N) arrays. Paths can then be constructed using `make_path`.
        '''
        
        if source is None:  # All sources
            self.weight_matrix = nx.to_numpy_array(self.G, weight='weight')
            self.distances, self.predecessors = SP(self.weight_matrix, method='D', directed=False, return_predecessors=True)
                
        else:  # Specific sources
            source = np.atleast_1d(source)  # Ensure source is an array
            for s in source:
                self.distances[s], self.paths[s] = nx.single_source_dijkstra(self.G, s)
                for t in range(self.N):
                    if t not in self.distances[s]:  # If there is no path from s to t
                        self.distances[s][t] = np.inf
                        self.paths[s][t] = None
        return self

    def path(self, endpoints:tuple|ndarray[int]):
        '''
        Returns paths from `endpoints`.
        If paths were already computed then simply retrieve them, else reconstruct them from `self.predecessor` and store them.
        '''
        if endpoints is None : return
        endpoints = np.atleast_2d(endpoints)  # Ensure `endpoints` is a (n, 2) array of endpoints.
        paths = []
        for i, j in endpoints:
            if i in self.paths and j in self.paths[i]:  # If path already computed, retrieve it
                paths.append(self.paths[i][j])
            else:  # Else reconstruct the path from predecessors
                assert hasattr(self, 'predecessors'), "Path cannot be computed. `shortest_path` needs to be called first with the right source(s)."
                p = [j]  # List of intermediate points making up the path
                k = j
                try:
                    while k != i:  # While not reaching the starting point
                        k = self.predecessors[i, k]
                        p.append(k)
                    paths.append(np.array(p[::-1]))  # Invert the path to go from i to j
                except:  # Case where there is no path between i and j
                    paths.append(None)
                # Add path to self.paths
                if i in self.paths : self.paths[i][j] = paths[-1]
                else : self.paths[i] = {j : paths[-1]}
        return paths
    
    def make_path(self, *args):
        '''
        Shortcut for making and storing paths then returning self instead of the path.
        Takes the same arguments as `path` but returns self.
        '''
        self.path(*args)
        return self



## Particular metrics

@dataclass
class DTM_arg:
    '''Class to store DTM parameters.'''
    m : float
    p : int = 2
    beta : float = 1

class DTM(DistanceToMeasure):
    def __init__(self, X:ndarray[float, (...,...)], dtm_arg:DTM_arg, dtm_precomputed:Callable=None, **kwargs):
        '''
        Creates the DTM of the empirical measure associated with point cloud `X`.
        If `dtm_precomputed` is provided, then we skip calling `DistanceToMeasure` and use that instead.
        `dtm_precomputed` must have signature `(ndarray[float, (...,...)], DTM_arg -> ndarray[float, (...,)])`.
        '''
        if dtm_precomputed is None:
            k = int(dtm_arg.m*len(X))+1  # Number of neighbors
            super().__init__(min(k, len(X)), dtm_arg.p)  # Initialize the DTM
            self.fit(X)
            self.precomputed = None
        else:
            try:
                self.precomputed = lambda X : dtm_precomputed(X, dtm_arg)
            except:  # In case dtm_precomputed is not vectorized
                self.precomputed = np.vectorize(lambda X : dtm_precomputed(X, dtm_arg), signature='(d)->()')
        
        self.dtm_arg = dtm_arg
        self.__dict__.update(dtm_arg.__dict__)  # Copy dtm_arg fields
        self.X = X
        self.n = len(X)
        return

    def transform(self, Y:ndarray[float, (...,...)]):
        '''New 'transform' method that uses the original but with `beta` supports.'''
        if self.precomputed is not None : return self.precomputed(Y)
        else : return super().transform(Y) ** self.beta  # Use original `transform` method
    

class FDTM(Metric):
    '''
    Fermat-DTM metric over point cloud `X`.
    If `dtm` is a DTM_arg then the corresponding DTM associated to `X` is computed.
    Else `dtm` is the precomputed DTM.
    `kwargs` are passed on to `DTM`.
    '''
    def __init__(self, X:ndarray[float, (...,...)], dtm:DTM_arg | DTM, **kwargs):
        super().__init__(X)
        if isinstance(dtm, DTM_arg):
            self.dtm = DTM(X, dtm, **kwargs)
        else:
            self.dtm = dtm
        self.dtm_array = self.dtm.transform(X)  # DTM values at points in X
    
    def f_edge_all(self, precision_segment=2, **kwargs):
        '''
        Computes the edges from all endpoints in `self.X`, as the average of `self.dtm` at endpoints times the Euclidean distance.
        The edge value is the average of the endpoints' dtm.
        '''
        L = pairwise_dist(self.X)
        if precision_segment == 2:
            D_i = self.dtm_array[:, np.newaxis]  # Shape (N, 1)
            D_j = self.dtm_array[np.newaxis, :]  # Shape (1, N)
            return L * (D_i + D_j)
    
        else:
            # First create an array of all edges 
            N, d = self.X.shape
            i, j = np.triu_indices(N, k=1)
            t = np.linspace(0, 1, precision_segment)
            Pi = self.X[i][:, None, :]
            Pj = self.X[j][:, None, :]
            P = Pi * (1 - t)[None, :, None] + Pj * t[None, :, None]  # P has shape (N_pairs, precision, d)
            N_pairs = P.shape[0]  # Number of pairs i < j
            
            dtm_edges = self.dtm.transform(P.reshape(N_pairs*precision_segment, d)).reshape(N_pairs, precision_segment)  # DTM needs reshaping

            w = np.ones(precision_segment)
            w[0], w[-1] = 0.5, 0.5
            average_edges = (dtm_edges * w).sum(axis=1) / (precision_segment - 1)  # Weighted average

            # Put back the weights from a (N_pairs,) array to a (N, N) array
            weights = np.zeros((N, N))
            weights[i, j] = average_edges
            weights[j, i] = average_edges
            return L * weights
    

    def f_edge(self, i:int|ndarray[int, (...,)], j:int|ndarray[int, (...,)], precision_segment:int=2, **kwargs):
        '''
        Computes the edges with first endpoints indices `i` and second endpoints indices `j`, as the average of `self.dtm` over a regular subdivision of the segment of `precision_segment` points.
        `i` and `j` are indices or same-size arrays of indices for points in `self.X`.
        The edge value is the average of the endpoints' dtm.
        '''
        assert precision_segment >= 2
        L = Euclidean_dist(self.X, i, j)
        
        if isinstance(i, int):
            i = np.array([i])
            j = np.array([j])

        # First create an array of all edges 
        N, d = self.X.shape
        t = np.linspace(0, 1, precision_segment)
        Pi = self.X[i][:, None, :]
        Pj = self.X[j][:, None, :]
        P = Pi * (1 - t)[None, :, None] + Pj * t[None, :, None]  # P has shape (N_pairs, precision, d)
        N_pairs = P.shape[0]  # Number of pairs i < j
        
        dtm_edges = self.dtm.transform(P.reshape(N_pairs*precision_segment, d)).reshape(N_pairs, precision_segment)  # DTM needs reshaping

        w = np.ones(precision_segment)
        w[0], w[-1] = 0.5, 0.5
        return L * (dtm_edges * w).sum(axis=1) / (precision_segment - 1)  # Weighted average
    

class Euclidean(Metric):
    def __init__(self, X:ndarray[float, (...,...)]):
        '''Euclidean metric'''
        super().__init__(X)
        
    def f_edge_all(self, **kwargs):
        return pairwise_dist(self.X)
    
    def f_edge(self, i:int|ndarray[int, (...,)], j:int|ndarray[int, (...,)], **kwargs):
        return Euclidean_dist(self.X, i, j)
    

class Fermat(Metric):
    def __init__(self, X:ndarray[float, (...,...)], alpha:float):
        '''Classic Fermat metric'''
        assert alpha >= 1, "alpha must be at least 1."
        super().__init__(X)
        self.alpha = alpha
        
    def f_edge_all(self, **kwargs):
        return pairwise_dist(self.X)**self.alpha
    
    def f_edge(self, i:int|ndarray[int, (...,)], j:int|ndarray[int, (...,)], **kwargs):
        return Euclidean_dist(self.X, i, j)**self.alpha