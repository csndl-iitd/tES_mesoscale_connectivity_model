# processing.py (ES_Adaptive class definition)
import numpy as np
import networkx as nx

import sys
sys.path.append('../Model/tES/')
sys.path.append('./MouseBrainLib/')

# wrapper.py (Wrapper class and network creation)
import networkx as nx
from tES_Adaptive import tES_Adaptive
from DataUtils import DataUtils

class MBN_RC(tES_Adaptive, DataUtils):
    
    def __init__(self, 
                 nepochs=10000, 
                 dt=0.01, 
                 lambda_o=0.01, 
                 alpha=0.01,
                 beta=0.002,
                 plot_bifurcation=False, 
                 epochs_per_lambda_o=10000, 
                 step_size_lambda_o=0.003):
        
        DataUtils.__init__(self)
        
        self.nepochs = nepochs
        self.dt = dt
        self.lambda_o = lambda_o
        self.plot_bifurcation = plot_bifurcation
        self.epochs_per_lambda_o = epochs_per_lambda_o
        self.step_size_lambda_o = step_size_lambda_o
        self.alpha = alpha
        self.beta = beta
        
        self.initialize_mbn_network()
        
        tES_Adaptive.__init__(self, 
                              self.A, 
                              self.N, 
                              self.lambda_o, 
                              self.alpha,
                              self.beta,
                              self.nepochs, 
                              self.dt, 
                              self.plot_bifurcation, 
                              self.epochs_per_lambda_o, 
                              self.step_size_lambda_o)

    def initialize_mbn_network(self):
        self.N = self.N_REGIONS_WHOLE_BRAIN
        self.A = np.zeros([self.N, self.N]) 
        self.A[:] = self.WHOLE_BRAIN_CONN 
        self.A = np.mat(self.A)
        
        # Create graph object from adjacency matrix
        self.G = nx.from_numpy_array(self.A)
        self.avg_degree = (sum(dict(self.G.degree()).values())/self.N)
        self.A = np.mat(nx.adjacency_matrix(self.G).todense())
        np.fill_diagonal(self.A, 0)
        
        print("Average shortest path length: ", nx.average_shortest_path_length(self.G))
        print("Avg clustering coefficient: ", nx.average_clustering(self.G))

    def run_model(self):
        tES_Adaptive.run_model(self)