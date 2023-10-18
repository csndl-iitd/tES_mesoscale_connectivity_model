# processing.py (ES_Adaptive class definition)
import numpy as np
import networkx as nx

import sys
sys.path.append('../Model/ES/')

# wrapper.py (Wrapper class and network creation)
import networkx as nx
from ES_Adaptive import ES_Adaptive

class SWN_No_RC(ES_Adaptive):
    
    def __init__(self, 
                 nepochs=10000, 
                 dt=0.01, 
                 lambda_o=0.01, 
                 plot_hysteresis=False, 
                 epochs_per_lambda_o=10000, 
                 step_size_lambda_o=0.003):
        
        self.nepochs = nepochs
        self.dt = dt
        self.lambda_o = lambda_o
        self.plot_hysteresis = plot_hysteresis
        self.epochs_per_lambda_o = epochs_per_lambda_o
        self.step_size_lambda_o = step_size_lambda_o
        
        self.create_undirected_network()
        
        super().__init__(self.A, 
                         self.N, 
                         self.lambda_o, 
                         self.nepochs, 
                         self.dt, 
                         self.plot_hysteresis, 
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