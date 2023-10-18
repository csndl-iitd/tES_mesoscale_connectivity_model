import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

class tES_Adaptive:
    """
    This class works for both binary/weighted directed/undirected networks.
    """
    
    def __init__(self, 
                 adjacency_matrix, 
                 num_nodes, 
                 lambda_o, 
                 alpha=0.01,
                 beta=0.002,
                 num_epochs=20000, 
                 dt=0.01, 
                 plot_bifurcation=False, 
                 epochs_per_lambda_o=10000, 
                 step_size_lambda_o=0.003):
        """
        Initialize the ES_Adaptive class with the provided parameters.

        Parameters:
            - adjacency_matrix (numpy.ndarray): The adjacency matrix representing the network.
            - num_nodes (int): The number of nodes in the network.
            - lambda_o (float): The initial coupling strength.
            - num_epochs (int): The number of simulation epochs (default is 20000).
            - dt (float): The time step for simulation (default is 0.01).
            - plot_hysteresis (bool): Whether to vary lambda_o through adiabatic progression 
                                      (default is False).
            - epochs_per_lambda_o (int): The number of epochs before changing lambda_o 
                                         (default is 10000).
            - step_size_lambda_o (float): The step size for changing lambda_o (default is 0.003).
        """
        self.N = num_nodes
        self.NEPOCHS = num_epochs
        self.DT = dt
        self.A = np.mat(adjacency_matrix)
        self.LAMBDA_O = lambda_o
        self.ALPHA = alpha
        self.BETA = beta
        self.plot_bifurcation = plot_bifurcation
        self.epochs_per_lambda_o = epochs_per_lambda_o
        self.step_size_lambda_o = step_size_lambda_o
        
        self.prepare_matrices()
        self.initialize()
    
    def prepare_matrices(self):
        """
        Matrix can be directed/undirected and binary/weighted.
        
        Local order and degree always uses binary undirected matrix.
        """
        self.binary_conn = np.mat(np.zeros([self.N, self.N]))
        self.binary_conn[:] = np.mat(np.where(self.A > 0, 1, 0))
        
        self.UNDIRECTED_A = (self.binary_conn + self.binary_conn.T) / 2
        self.UNDIRECTED_A[:] = np.mat(np.where(self.UNDIRECTED_A > 0, 1, 0))
        
    def initialize(self):
        # Node's degree
        self.K = np.mat(np.zeros([self.N, 1]))
        self.K = np.sum(self.UNDIRECTED_A, axis=1)
        
        # Initialize phases and natural frequencies
        self.THETA = np.mat(np.random.uniform(-np.pi, np.pi, [self.N, 1]))
        self.NAT_FREQ = np.mat(np.random.uniform(-1, 1, [self.N, 1]))
        
        # Local Order, Global order, and adaptive coupling
        self.LOCAL_ORDER = np.mat(np.zeros([self.N, 1]))
        self.SetLocalOrder()
        self.SetGlobalOrder()
        
        self.ADAPTIVE_COUPLING = np.mat(np.zeros([self.N, 1]))
        self.ADAPTIVE_COUPLING = self.LOCAL_ORDER
        
        # Set resource baths
        self.LAMBDA = np.mat(np.zeros([self.N, 1]))

    def run_model(self):
        self.GLOBAL_ORDER_VERBOSE = np.zeros(self.NEPOCHS)
        self.LAMBDA_ = np.zeros([self.N, self.NEPOCHS])
        self.LAMBDA_ = []
        self.LAMBDA_O_ = []

        for i in range(self.NEPOCHS):
            
            if self.plot_bifurcation:
                
                if (i % self.epochs_per_lambda_o == 0): 
                    self.LAMBDA_O = self.LAMBDA_O + self.step_size_lambda_o
                    print(f'\nLAMBDA_O={self.LAMBDA_O}, Global Order (R)={self.GLOBAL_ORDER}\n')
            
            if i%2000 == 0:
                print(f'LAMBDA_O={self.LAMBDA_O}, Global Order(R)={self.GLOBAL_ORDER}')
            
            wts = np.sum(np.multiply(self.A.T, np.sin(self.THETA.T - self.THETA)), axis=1)
            coupling = np.multiply(self.LAMBDA, np.multiply(self.ADAPTIVE_COUPLING, wts))
            dTHETA = (self.NAT_FREQ + coupling) * self.DT
            dLAMBDA = (self.ALPHA*(self.LAMBDA_O - self.LAMBDA) -
                       self.BETA*self.LOCAL_ORDER) * self.DT

            self.THETA = self.THETA + dTHETA
            self.LAMBDA = self.LAMBDA + dLAMBDA
            
            self.SetLocalOrder()
            self.SetGlobalOrder()

            # Update adaptive coupling weight
            self.ADAPTIVE_COUPLING[:] = self.LOCAL_ORDER
            self.GLOBAL_ORDER_VERBOSE[i] = self.GLOBAL_ORDER
            self.LAMBDA_.append(self.LAMBDA)
            self.LAMBDA_O_.append(self.LAMBDA_O)
            
        print(self.GLOBAL_ORDER)

    def SetLocalOrder(self):
        E = np.exp(1j * self.THETA)
        K_ = 1 / self.K
        K_[K_ == np.inf] = 0
        R = np.multiply(K_, np.matmul(self.UNDIRECTED_A, E))
        self.LOCAL_ORDER = np.absolute(R)

    def SetGlobalOrder(self):
        E = np.exp(1j * self.THETA)
        R = np.sum(E) / self.N
        self.GLOBAL_ORDER = np.absolute(R)

    def GetLocalOrder(self, theta):
        E = np.exp(1j * theta)
        K_ = 1 / self.K
        K_[K_ == np.inf] = 0
        R = np.multiply(K_, np.matmul(self.UNDIRECTED_A, E))
        return np.absolute(R), np.angle(R)
