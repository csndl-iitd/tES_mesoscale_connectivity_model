# processing.py (ES_Adaptive class definition)
import numpy as np
import networkx as nx

import sys
sys.path.append('../Model/tES/')

# wrapper.py (Wrapper class and network creation)
import networkx as nx
from tES_Adaptive import tES_Adaptive

class SWN_RC(tES_Adaptive):
    
    def __init__(self, 
                 nepochs=10000, 
                 dt=0.01, 
                 lambda_o=0.01, 
                 alpha=0.01,
                 beta=0.002,
                 plot_bifurcation=False, 
                 epochs_per_lambda_o=10000, 
                 step_size_lambda_o=0.003):
        
        self.nepochs = nepochs
        self.dt = dt
        self.lambda_o = lambda_o
        self.plot_bifurcation = plot_bifurcation
        self.epochs_per_lambda_o = epochs_per_lambda_o
        self.step_size_lambda_o = step_size_lambda_o
        self.alpha = alpha
        self.beta = beta
        
        self.create_undirected_network()
        
        super().__init__(self.A, 
                         self.N, 
                         self.lambda_o, 
                         self.alpha,
                         self.beta,
                         self.nepochs, 
                         self.dt, 
                         self.plot_bifurcation, 
                         self.epochs_per_lambda_o, 
                         self.step_size_lambda_o)

    def create_undirected_network(self):
        """
        Create an undirected graph using Watts-Strogatz model
        """
        self.N = 400
        self.G = nx.watts_strogatz_graph(n=self.N, k=45, p=0.232)
        self.avg_degree = (sum(dict(self.G.degree()).values()) / self.N)
        self.A = np.mat(nx.adjacency_matrix(self.G).todense())

    def run_model(self):
        super().run_model()