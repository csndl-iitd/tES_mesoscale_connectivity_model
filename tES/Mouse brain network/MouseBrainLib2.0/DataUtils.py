"""
TIP: 
    1. Never use  [[0]*3]*3 for array assignment - This creates three duplicate list. Check out below code to view the weird result it gives:
        X = [[2]*3]*2
        X[0][0] = 3
        print(X)
        #Extremely weired!!! This is assigning to all the coloumn 0th element to 3.
        
        TODO:
        Reconstruct between -1 and 1
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from glob import glob
from os import path
import scipy
from scipy import signal
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
# import community as community_louvain
from community import community_louvain
import networkx as nx
import matplotlib.patches as patches
import matplotlib.colors as mcolors

class DataUtils:
    
    """
    Variables holding data:
        1. self.exp
        2. self.eegdata (EEG data for all 30 channels)
        3. self.processedData (filtered EEg signal)
        4. self.segmented_data (EEG signal cut into 3s intervals)
        5. self.ERP (ERP for each of each of the stimulus group)
    """
    
    #Constants
    # INTER_REGION_CONN_FILE = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/inter-region-p-value-filtered-labelled.csv'
    # INTER_REGION_PVAL_FILE = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/regions-p-values - unlabelled.csv'
    
    INTER_REGION_CONN_FILE = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/inter_region_conn_filtered.csv'
    INTER_REGION_PVAL_FILE = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/inter_region_p_value_filtered.csv'
    
    INTER_REGION_CONN_FILE_IPSI = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/connectivity matrices/connectivity matrices/ipsilateral/ipsilateral_original.csv'
    INTER_REGION_PVAL_FILE_IPSI = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/connectivity matrices/connectivity matrices/ipsilateral/ipsilateral_original_p_val_unlabelled.csv'
    INTER_REGION_DIST_FILE_IPSI = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/connectivity matrices/connectivity matrices/ipsilateral/Ipsilateral_InterRegionDistances.csv'
    
    INTER_REGION_CONN_FILE_CONTRA = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/connectivity matrices/connectivity matrices/contralateral/contralateral_original.csv'
    INTER_REGION_PVAL_FILE_CONTRA = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/connectivity matrices/connectivity matrices/contralateral/contralateral_original_p_val_unlabelled.csv'
    INTER_REGION_DIST_FILE_CONTRA = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/connectivity matrices/connectivity matrices/contralateral/Contralateral_InterRegionDistances.csv'
    
    MB_WB_REGION_COMMUNITY_FILE = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/mb_communities.npz'
    MB_WB_REGION_COMMUNITY_DICT_FILE = '/home/jupyter-avinash/ranjan_env_15_Feb_2023/Explosive Synchronization/Mouse brain/MouseBrainLib/mb_communities_dict.pickle'
    
    MB_CONN_PVAL_THR = 0.05 #0.0699
    
    DATASTORE = '/home/jupyter-avinash/datastore/allen_mouse_eeg'
    MOUSE = 'mouse599975'
    EXP = 'estim_vis_2022-03-31_12-03-06'
    STIM_DATA = DATASTORE + '/' + MOUSE + '/' + EXP + '/experiment1/recording1/all_stim_log.csv'
    SAMPLING_FREQ = 500 #Sampling frequency
    NOTCH_FLT_FREQ = 30 #Notch filter frequency
    AC_FREQ = 50        #Frequency to remove
    LPF_FC = 100        #Cutoff frequency for low pass filter
    #TODO: This has to be modified for whole-brain connectivity
    REGIONS = {
                'ISOCORTEX': {'idx': 0, 'num_regions': 38}, #cortical
                'OLF': {'idx': 38, 'num_regions': 11},      #olfactory - Limbic system
                'HPF': {'idx': 49, 'num_regions': 11},      #Hippocampus - Limbic system
                'CTXsp': {'idx': 60, 'num_regions': 7},     #cerebral plate : Amygdla - limbic system
                'STR': {'idx': 67, 'num_regions': 12},      #Striatum - basalganglia
                'PAL': {'idx': 79, 'num_regions': 8},       #Pallidum - basalganglia
                'TH': {'idx': 87, 'num_regions': 35},       #Thalamus
                'HY': {'idx': 122, 'num_regions': 20},      #Hypothalamus - Limbic system
                'MB': {'idx': 142, 'num_regions': 21},      #Midbrain
                'P': {'idx': 163, 'num_regions': 13},       #Pons
                'MY': {'idx': 176, 'num_regions': 25},      #Medulla 
                'CB': {'idx': 201, 'num_regions': 12},      #Cerebellum
                'ALL': {'idx': 0, 'num_regions': 213}
            }
    
    """
    Cortex: ISOCORTEX, OLF, CTXsp
    Hippocampus: HPF
    Striatum: STR, PAL
    Diencephalon: TH, HY
    Midbrain: MB
    Hindbrain: P, MY, CB
    """
    
    INDEX_GROUPS = {
        range(38): 'ISOCORTEX', #cortical
        range(38, 49): 'OLF',      #olfactory - Limbic system
        range(49, 60): 'HPF',      #Hippocampus - Limbic system
        range(60, 67): 'CTXsp',     #cerebral plate : Amygdla - limbic system
        range(67, 79): 'STR',      #Striatum - basalganglia
        range(79, 87): 'PAL',       #Pallidum - basalganglia
        range(87, 122): 'TH',       #Thalamus
        range(122, 142): 'HY',      #Hypothalamus - Limbic system
        range(142, 163): 'MB',      #Midbrain
        range(163, 176): 'P',       #Pons
        range(176, 201): 'MY',      #Medulla 
        range(201, 213): 'CB',  
        range(213, 251): 'ISOCORTEX', #cortical
        range(251, 262): 'OLF',      #olfactory - Limbic system
        range(262, 273): 'HPF',      #Hippocampus - Limbic system
        range(273, 280): 'CTXsp',     #cerebral plate : Amygdla - limbic system
        range(280, 292): 'STR',      #Striatum - basalganglia
        range(292, 300): 'PAL',       #Pallidum - basalganglia
        range(300, 335): 'TH',       #Thalamus
        range(335, 355): 'HY',      #Hypothalamus - Limbic system
        range(355, 376): 'MB',      #Midbrain
        range(376, 389): 'P',       #Pons
        range(389, 414): 'MY',      #Medulla 
        range(414, 426): 'CB', 
    }
    
    def PlotDistribution(self):
        data = self.GetRegionalConnectivityMatrix('ALL', 'ALL')
        # Plot the distribution of connections using a histogram
        log_data = np.log10(data[data > 1e-7])
        plt.hist(log_data.flatten(), bins=100)
        plt.xlabel('Connection Strength')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ipsilateral Connections')
        plt.show()
    
    def Dummy(self):
        # Define range of values
        lower_bound = 10**(-2.5)
        upper_bound = 10**(-1.5)

        # Filter row and column indices based on the range of values
        row_indices, col_indices = np.where((self.INTR_RGN_CONN >= lower_bound) & (self.INTR_RGN_CONN <= upper_bound))
        
        # Plot joint distribution of row and column indices
        plt.hist2d(row_indices, col_indices, bins=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Row Index')
        plt.ylabel('Column Index')
        plt.title('Joint Distribution of Row and Column Indices')
        plt.show()

    def GetRegionalConnectivityMatrix(self, src, target):
        src_idx = DataUtils.REGIONS[src]['idx']
        src_num_regions = DataUtils.REGIONS[src]['num_regions']
        target_idx = DataUtils.REGIONS[target]['idx']
        target_num_regions = DataUtils.REGIONS[target]['num_regions']
        
        return self.INTR_RGN_CONN[
            src_idx:(src_idx+src_num_regions), 
            target_idx:(target_idx+target_num_regions)]
    
    def ScaleRegionalConnectivity(self, src, target, factor):
        src_idx = DataUtils.REGIONS[src]['idx']
        src_num_regions = DataUtils.REGIONS[src]['num_regions']
        target_idx = DataUtils.REGIONS[target]['idx']
        target_num_regions = DataUtils.REGIONS[target]['num_regions']
        
        mat = self.GetRegionalConnectivityMatrix(src, target)
        print( src + ":" + target + " : ", np.count_nonzero(mat) )
        
        # Scale up connectivity within src and target
        self.INTR_RGN_CONN[src_idx:(src_idx+src_num_regions), 
                           target_idx:(target_idx+target_num_regions)] *= factor
    
    def GetIpsiConnectivityMatrix(self, src, target):
        src_idx = DataUtils.REGIONS[src]['idx']
        src_num_regions = DataUtils.REGIONS[src]['num_regions']
        target_idx = DataUtils.REGIONS[target]['idx']
        target_num_regions = DataUtils.REGIONS[target]['num_regions']
        
        return self.INTR_RGN_CONN_IPSI[
            src_idx:(src_idx+src_num_regions), 
            target_idx:(target_idx+target_num_regions)]
    
    def GetContraConnectivityMatrix(self, src, target):
        src_idx = DataUtils.REGIONS[src]['idx']
        src_num_regions = DataUtils.REGIONS[src]['num_regions']
        target_idx = DataUtils.REGIONS[target]['idx']
        target_num_regions = DataUtils.REGIONS[target]['num_regions']
        
        return self.INTR_RGN_CONN_CONTRA[
            src_idx:(src_idx+src_num_regions), 
            target_idx:(target_idx+target_num_regions)]
    
    def ScaleWBConnectivity(self, src, target, factor):
        src_idx = DataUtils.REGIONS[src]['idx']
        src_num_regions = DataUtils.REGIONS[src]['num_regions']
        target_idx = DataUtils.REGIONS[target]['idx']
        target_num_regions = DataUtils.REGIONS[target]['num_regions']
        
        mat_ipsi = self.GetIpsiConnectivityMatrix(src, target)
        print( "Ipsilateral: " + src + ":" + target + " : ", np.count_nonzero(mat_ipsi) )
        
        # Scale up connectivity within src and target
        self.INTR_RGN_CONN_IPSI[src_idx:(src_idx+src_num_regions), 
                           target_idx:(target_idx+target_num_regions)] *= factor
        
        mat_contra = self.GetContraConnectivityMatrix(src, target)
        print( "Contralateral: " + src + ":" + target + " : ", np.count_nonzero(mat_contra) )
        
        # Scale up connectivity within src and target
        self.INTR_RGN_CONN_CONTRA[src_idx:(src_idx+src_num_regions), 
                           target_idx:(target_idx+target_num_regions)] *= factor
        
        # Initialize WB connectivity again
        self.init_whole_brain_connectivity()
        
    def __init__(self):    
        self.init_inter_region_connectivity()
        self.init_inter_region_connectivity_ipsi()
        self.init_inter_region_connectivity_contra()
        self.init_whole_brain_connectivity()
        self.form_communities()
        self.init_region_id()
        self.calculate_in_and_out_degree()
        print("test")
    
    def init_region_id(self):
        # Define the distinct region strings
        self.WB_ANATOMICAL_AREAS = np.asarray(['ISOCORTEX', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CB'])

        # Define the corresponding colors for each region
        self.WB_ANATOMICAL_AREAS_COLOR = np.asarray(['red', 'red', 'green', 'red', 'magenta', 'magenta', 'orange', 'orange', 'purple', 'brown', 'brown', 'brown'])
        
        """
        Cortex: ISOCORTEX, OLF, CTXsp
        Hippocampus: HPF
        Striatum: STR, PAL
        Diencephalon: TH, HY
        Midbrain: MB
        Hindbrain: P, MY, CB
        """
        self.WB_AREAS_ORDERED_BY_COARSER_GROUP = ['ISOCORTEX', 'OLF', 'CTXsp', 'HPF', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CB']
        self.WB_AREAS_COLOR_ORDERED_BY_COARSER_GROUP = ['red', 'red', 'red', 'green', 'magenta', 'magenta', 'orange', 'orange', 'purple', 'brown', 'brown', 'brown']
        # self.COARSER_WB_ANATOMICAL_AREAS = np.asarray(['Cortex', 
        #                                                'Hippocampus', 
        #                                                'Striatum', 
        #                                                'Diencephalon', 
        #                                                'Midbrain', 
        #                                                'Hindbrain'])
        
        self.COARSER_WB_ANATOMICAL_AREAS = np.asarray(['CRTX', 
                                                       'HIPP', 
                                                       'STR', 
                                                       'TH', #Diencephalon
                                                       'MID', 
                                                       'HIND'])

        # Define the corresponding colors for each region
        self.COARSER_WB_ANATOMICAL_AREAS_COLOR = np.asarray(['red', 'blue', 'green', 
                                                             'orange', 'purple', 'brown'])
        
        
        self.WB_ANATOMICAL_AREAS_COUNT = len(self.WB_ANATOMICAL_AREAS)
        self.COARSER_WB_ANATOMICAL_AREAS_COUNT = len(self.COARSER_WB_ANATOMICAL_AREAS)
        
        # self.WB_REGIONS_CNT_SINGLE_HEMI = np.asarray([38, 11, 11, 7, 12, 8, 35, 20, 21, 13, 25, 12])
        self.NODE_COUNT_PER_AREA = np.asarray([56, 11, 20, 55, 21, 50])
    
        self.NODE_ANATOMICAL_AREA_ID = np.empty(self.N_REGIONS_WHOLE_BRAIN, dtype='U10')
        self.NODE_ANATOMICAL_AREA_COLOR = np.empty(self.N_REGIONS_WHOLE_BRAIN, dtype='U10')
        
        for nodes, region in self.INDEX_GROUPS.items():
            self.NODE_ANATOMICAL_AREA_ID[nodes] = region
            self.NODE_ANATOMICAL_AREA_COLOR[nodes] = self.WB_ANATOMICAL_AREAS_COLOR[
                np.where(self.WB_ANATOMICAL_AREAS==region)]
    
    def calculate_in_and_out_degree(self):
        G = nx.from_numpy_array(self.WHOLE_BRAIN_CONN, create_using=nx.DiGraph())
        
        # Calculate in-degree as a dictionary
        in_degree_dict = dict(G.in_degree())
        # Convert in-degree dictionary values to an array
        self.IN_DEGREE = np.array(list(in_degree_dict.values()))
        
        # Calculate in-degree as a dictionary
        out_degree_dict = dict(G.out_degree())
        # Convert in-degree dictionary values to an array
        self.OUT_DEGREE = np.array(list(out_degree_dict.values()))
        
    def form_communities(self):
        if os.path.isfile(DataUtils.MB_WB_REGION_COMMUNITY_FILE):
            print(f"{self.MB_WB_REGION_COMMUNITY_FILE} already exists")
            data = np.load(DataUtils.MB_WB_REGION_COMMUNITY_FILE)
            # access the data using the keys in the npz file
            self.ORDERED_NODES = data['arr1']
            self.N_COMMUNITIES = data['arr2']
            
            # load the dictionary from the file
            with open(DataUtils.MB_WB_REGION_COMMUNITY_DICT_FILE, 'rb') as f:
                self.COMMUNITIES = pickle.load(f)
            
            self.COLOR_MAP = plt.cm.get_cmap('rainbow', self.N_COMMUNITIES)
            
            return
        else:
            print(f"{self.MB_WB_REGION_COMMUNITY_DICT_FILE} does not exist")
        
        self.G = nx.from_numpy_array(self.WHOLE_BRAIN_CONN) #create_using=nx.DiGraph()
        partition = community_louvain.best_partition(self.G, resolution=1)
        self.N_COMMUNITIES = len(set(partition.values()))
        self.COLOR_MAP = plt.cm.get_cmap('rainbow', self.N_COMMUNITIES)
        
        self.ORDERED_NODES = []
        self.COMMUNITIES = {}
        for node, community_id in partition.items():
            if community_id not in self.COMMUNITIES:
                self.COMMUNITIES[community_id] = [node]
            else:
                self.COMMUNITIES[community_id].append(node)
        
        for community_id, nodes in self.COMMUNITIES.items():
            self.ORDERED_NODES.extend(nodes)
    
        # Save the dictionary to a file
        with open(DataUtils.MB_WB_REGION_COMMUNITY_DICT_FILE, 'wb') as f:
            pickle.dump(self.COMMUNITIES, f)
        
        np.savez(DataUtils.MB_WB_REGION_COMMUNITY_FILE, arr1=np.asarray(self.ORDERED_NODES), 
                 arr2=np.array(self.N_COMMUNITIES))
        
    def init_inter_region_connectivity(self):
        file = open(DataUtils.INTER_REGION_CONN_FILE)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        #Connectivity matrix normalized between 0 and 1
        self.LABELS = rows[0,:]
        self.INTR_RGN_CONN = (rows[1:,:]).astype('float')
        self.N_REGIONS = len(self.INTR_RGN_CONN)
        
        file = open(DataUtils.INTER_REGION_PVAL_FILE)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        self.P_VAL = rows.astype('float')
    
    def init_inter_region_connectivity_ipsi(self):
        """ Filter connection strength """
        file = open(DataUtils.INTER_REGION_CONN_FILE_IPSI)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        #Connectivity matrix normalized between 0 and 1
        self.LABELS_IPSI = rows[0,1:]
        self.INTR_RGN_CONN_IPSI = (rows[1:,1:]).astype('float')
        self.N_REGIONS_IPSI = len(self.INTR_RGN_CONN_IPSI)
        
        """ Filter all the p-values """
        file = open(DataUtils.INTER_REGION_PVAL_FILE_IPSI, encoding='utf-8-sig')
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        self.P_VAL_IPSI = rows.astype('float')
        
        """ Filter all the region distances """
        file = open(DataUtils.INTER_REGION_DIST_FILE_IPSI)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        self.INTR_RGN_DIST_IPSI = (rows[1:,1:]).astype('float')
        # Convert all distances in mm
        self.INTR_RGN_DIST_IPSI = self.INTR_RGN_DIST_IPSI*0.001 
        
        """ Filter significant connections only """
        mask = self.P_VAL_IPSI > DataUtils.MB_CONN_PVAL_THR
        self.INTR_RGN_CONN_IPSI[mask] = 0
        # self.INTR_RGN_DIST_IPSI[mask] = 0
        
    def init_inter_region_connectivity_contra(self):
        """ Filter connection strength """
        file = open(DataUtils.INTER_REGION_CONN_FILE_CONTRA)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        #Connectivity matrix normalized between 0 and 1
        self.LABELS_CONTRA = rows[0,1:]
        self.INTR_RGN_CONN_CONTRA = (rows[1:,1:]).astype('float')
        self.N_REGIONS_CONTRA = len(self.INTR_RGN_CONN_CONTRA)
        
        """ Filter all the p-values """
        file = open(DataUtils.INTER_REGION_PVAL_FILE_CONTRA, encoding='utf-8-sig')
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        self.P_VAL_CONTRA = rows.astype('float')
        
        """ Filter all the region distances """
        file = open(DataUtils.INTER_REGION_DIST_FILE_CONTRA)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        self.INTR_RGN_DIST_CONTRA = (rows[1:,1:]).astype('float')
        # Convert all distances in mm
        self.INTR_RGN_DIST_CONTRA = self.INTR_RGN_DIST_CONTRA*0.001
        
        # Filter significant connections only
        mask = self.P_VAL_CONTRA > DataUtils.MB_CONN_PVAL_THR
        self.INTR_RGN_CONN_CONTRA[mask] = 0
        # self.INTR_RGN_DIST_CONTRA[mask] = 0
    
    def init_whole_brain_connectivity(self):
        self.N_REGIONS_WHOLE_BRAIN = self.N_REGIONS_IPSI + self.N_REGIONS_CONTRA
        self.WHOLE_BRAIN_CONN = np.zeros([self.N_REGIONS_WHOLE_BRAIN, self.N_REGIONS_WHOLE_BRAIN])
        self.WHOLE_BRAIN_REGN_DISTANCES = np.zeros([self.N_REGIONS_WHOLE_BRAIN, self.N_REGIONS_WHOLE_BRAIN])
        
        """ Create connection strength matrix """
        top_row = np.block([[self.INTR_RGN_CONN_IPSI, self.INTR_RGN_CONN_CONTRA]])
        bottom_row = np.block([[self.INTR_RGN_CONN_CONTRA, self.INTR_RGN_CONN_IPSI]])
        self.WHOLE_BRAIN_CONN[:] = np.concatenate((top_row, bottom_row))
        
        # Scale whole brain matrix between 0 and 1
        # Create a scaler object
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Flatten the matrix for scaling
        scaled_matrix = scaler.fit_transform(self.WHOLE_BRAIN_CONN.flatten().reshape(-1, 1))

        # Reshape the scaled matrix back to its original shape
        scaled_matrix = scaled_matrix.reshape(self.WHOLE_BRAIN_CONN.shape)
        
        self.WHOLE_BRAIN_CONN[:] = scaled_matrix
        
        """ Create p-value matrix """
        top_row = np.block([[self.P_VAL_IPSI, self.P_VAL_CONTRA]])
        bottom_row = np.block([[self.P_VAL_CONTRA, self.P_VAL_IPSI]])
        self.WHOLE_BRAIN_P_VAL = np.concatenate((top_row, bottom_row))
        
        """ Create distance matrix """
        top_row = np.block([[self.INTR_RGN_DIST_IPSI, self.INTR_RGN_DIST_CONTRA]])
        bottom_row = np.block([[self.INTR_RGN_DIST_CONTRA, self.INTR_RGN_DIST_IPSI]])
        self.WHOLE_BRAIN_REGN_DISTANCES = np.concatenate((top_row, bottom_row))
        
        self.LABELS_WB = np.concatenate((self.LABELS_IPSI, self.LABELS_CONTRA))
    
    def scale_connectivity_matrix_between_0_1(self):
        # Scale whole brain matrix between 0 and 1
        # Create a scaler object
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Flatten the matrix for scaling
        scaled_matrix = scaler.fit_transform(self.WHOLE_BRAIN_CONN.flatten().reshape(-1, 1))

        # Reshape the scaled matrix back to its original shape
        scaled_matrix = scaled_matrix.reshape(self.WHOLE_BRAIN_CONN.shape)
        
        self.WHOLE_BRAIN_CONN[:] = scaled_matrix
        
    def show_inter_region_connectivity(self):
        f = plt.figure()
        print(self.P_VAL.shape, self.INTR_RGN_CONN.shape)
        plt.imshow(self.INTR_RGN_CONN, alpha=(1-self.P_VAL))
        plt.title("Inter-region connectivity matrix (ipsi)")
        plt.show()
        
        f = plt.figure()
        print(self.P_VAL_CONTRA.shape, self.INTR_RGN_CONN_CONTRA.shape)
        plt.imshow(self.INTR_RGN_CONN_CONTRA, alpha=(1-self.P_VAL_CONTRA))
        plt.title("Inter-region connectivity matrix (contra)")
        plt.show()
        
    def show_inter_region_connectivity2(self):
        cmap = plt.cm.coolwarm
        
        brain_conn = np.log10(self.INTR_RGN_CONN_IPSI)
        brain_conn[brain_conn == -np.inf] = 0
        
        conn = np.copy(brain_conn)
        conn[conn < -3.5] = -3.5
        conn[conn > 1] = 1
        
        f = plt.figure()
        print(self.P_VAL_IPSI.shape, conn.shape)
        plt.imshow(conn, alpha=(1-self.P_VAL_IPSI), cmap=cmap)
        plt.title("Inter-region connectivity matrix (ipsi)")
        plt.colorbar()
        plt.show()
        
        cmap = plt.cm.coolwarm
        
        brain_conn = np.log10(self.INTR_RGN_CONN_CONTRA)
        brain_conn[brain_conn == -np.inf] = 0
        
        conn = np.copy(brain_conn)
        conn[conn < -3.5] = -3.5
        conn[conn > 1] = 1
        
        f = plt.figure()
        print(self.P_VAL_CONTRA.shape, conn.shape)
        plt.imshow(conn, alpha=(1-self.P_VAL_CONTRA), cmap=cmap)
        plt.title("Inter-region connectivity matrix (contra)")
        plt.colorbar()
        plt.show()
        
    def show_whole_brain_connectivity(self):
        whole_brain_conn = np.log10(self.WHOLE_BRAIN_CONN)
        whole_brain_conn[whole_brain_conn == -np.inf] = 0
        # print(whole_brain_conn)
        
        print(self.WHOLE_BRAIN_P_VAL.shape, self.WHOLE_BRAIN_CONN.shape)
        
        WHOLE_BRAIN_CONN_copy = np.copy(whole_brain_conn)
        WHOLE_BRAIN_CONN_copy[WHOLE_BRAIN_CONN_copy < -4.5] = -3.5
        WHOLE_BRAIN_CONN_copy[WHOLE_BRAIN_CONN_copy > 1] = 1

        # plt.imshow(self.WHOLE_BRAIN_CONN[DataUtils.ordered_nodes, :][:, DataUtils.ordered_nodes], 
        #            alpha=(1-self.WHOLE_BRAIN_P_VAL[DataUtils.ordered_nodes, :][:, DataUtils.ordered_nodes]))
        
        f = plt.figure()
        # fig, ax = plt.subplots(figsize=(12,8))
        cmap = plt.cm.coolwarm
        # Usung cmap here might not be a good idea, since this contains for both ipsi and contra
        plt.imshow(WHOLE_BRAIN_CONN_copy[self.ORDERED_NODES, :][:, self.ORDERED_NODES], alpha=(1-self.WHOLE_BRAIN_P_VAL[self.ORDERED_NODES, :][:, self.ORDERED_NODES]), 
                   cmap=cmap) #.reversed()
        plt.title("Whole brain connectivity")
        plt.colorbar()
        
        boxes_range = []
        prev_count = 0
        for community_id, nodes in self.COMMUNITIES.items():
            boxes_range.append((prev_count, prev_count + len(nodes) - 1))
            prev_count = prev_count + len(nodes)
                    
        for index, (community_id, nodes) in enumerate(self.COMMUNITIES.items()):
            box = patches.Rectangle((boxes_range[index][0], 
                                     boxes_range[index][0]), 
                                     boxes_range[index][1] - boxes_range[index][0], 
                                     boxes_range[index][1] - boxes_range[index][0], 
                                     linewidth=1.5, 
                                     edgecolor=self.COLOR_MAP(community_id), 
                                     facecolor='none')
            plt.gca().add_patch(box)
                    
#         # add labels for index groups to the x and y axis
#         yticks = [(k.start + k.stop)//2 for k in self.INDEX_GROUPS.keys()]
#         yticklabels = list(self.INDEX_GROUPS.values())
#         plt.yticks(yticks, yticklabels, fontsize=6)
        
#         xticks = [(k.start + k.stop)//2 for k in self.INDEX_GROUPS.keys()]
#         xticklabels = list(self.INDEX_GROUPS.values())
#         plt.xticks(xticks, xticklabels, rotation=90, fontsize=6)
        
        plt.show()
        
    def show_communities(self):
        # Create a color map for the communities
        # self.COLOR_MAP = plt.cm.get_cmap('rainbow', self.N_COMMUNITIES)
        
        # Create graph object from adjacency matrix
        self.A = np.zeros([self.N_REGIONS_WHOLE_BRAIN, self.N_REGIONS_WHOLE_BRAIN])
        self.G = nx.from_numpy_array(self.WHOLE_BRAIN_CONN)
        self.A = np.mat(nx.adjacency_matrix(self.G).todense())
        
        # Plot the adjacency matrix as a heatmap
        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(self.A, cmap='Greys', vmin=0, vmax=1)
        
        for community_id, nodes in self.COMMUNITIES.items():
            print(community_id)
            for idx_i, node_i in enumerate(nodes):
                for idx_j, node_j in enumerate(nodes):
                    ax.add_patch(plt.Rectangle((node_i-0.5, 
                                                node_j-0.5), 1, 1, fill=True, 
                                                edgecolor='none', 
                                                facecolor=self.COLOR_MAP(community_id)))
        
        
        # Add a color bar for the community legend
        cbar = plt.colorbar(im, ticks=[0, 1])
        cbar.ax.set_yticklabels(['0', '1'])
        cbar.ax.set_ylabel('Connection Strength')
        
        # add labels for index groups to the x and y axis
        yticks = [k.start for k in self.INDEX_GROUPS.keys()]
        yticklabels = list(self.INDEX_GROUPS.values())
        plt.yticks(yticks, yticklabels)
        
        xticks = [k.start for k in self.INDEX_GROUPS.keys()]
        xticklabels = list(self.INDEX_GROUPS.values())
        plt.xticks(xticks, xticklabels, rotation=90)
        
        # Show the plot
        plt.show()
    
    
"""
so_many_mouse = glob(datastore+"/mouse*")

for mouse in so_many_mouse:
    sessions = [path.basename(f) for f in glob(path.join(mouse, '*'))]
    print(sessions)
    
    for expt in sessions:
        if expt != 'histology':
            rec_folder = path.join(mouse, expt, 'experiment1/recording1')
            print(rec_folder)
            exp = EEGexp(rec_folder, preprocess=False, make_stim_csv=False)
            print(exp.ephys_params['EEG'])
"""