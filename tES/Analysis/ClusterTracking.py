import numpy as np
import random
from collections import deque

class SCTA:
    """
    Synchronization Cluster Tracking Algorithm
    """
    
    MAX_CLUSTER_ID = 1000
    MIN_CLUSTER_ID = 1
    
    def __init__(self, 
                 bin_conn, 
                 local_order,
                 local_order_phase,
                 sync_thrsh=0.6):
        """
        Parameters:
            - bin_conn (numpy.ndarray): binary directed MBN
            - local_order (numpy.ndarray): local order of all MNB nodes for one transient
        """
        self.bin_conn = bin_conn
        self.local_order = local_order
        self.local_order_phase = local_order_phase
        self.sync_thrsh = sync_thrsh
        
        self.window_size = local_order.shape[0]
        self.N = local_order.shape[1]
        self.used_numbers = set()
        self.all_numbers = set(range(1, SCTA.MAX_CLUSTER_ID))
        self.node_communities = np.zeros([self.window_size, self.N])
        self.count_per_sync_cluster = np.zeros([self.window_size,  SCTA.MAX_CLUSTER_ID])
    
    def get_unused_number(self):
        available_numbers = all_numbers - used_numbers
        if not available_numbers:
            raise ValueError("No more unused numbers available")
        number = random.choice(list(available_numbers))
        self.used_numbers.add(number)
        return number
    
    def clear_used_number_list(self):
        self.used_numbers.clear()

    def add_community_id_to_used_list(self, community_id):
        self.used_numbers.add(community_id)
    
    def get_central_node(self, t_idx, node_list, count):
        sorted_indices = np.argsort(self.local_order[t_idx, node_list])[::-1] 
        top_indices = sorted_indices[:count]

        # Get the corresponding nodes from the node_list
        central_nodes = [node_list[idx] for idx in top_indices]
        return central_nodes
    
    def track_clusters(self):
        thr = self.sync_thrsh
        NODE_COMMUNITIES = np.zeros([self.window_size, self.N]) 
        CENTRAL_NODES = []
        CENTRAL_NODES_CLUSTER_ID = []
        TIME_OF_ENTRY = np.zeros(self.N)

        # For verification purpose
        AVG_LOCAL_ORDER_PER_CLUSTER = np.zeros([self.window_size, SCTA.MAX_CLUSTER_ID])
        AVG_THETA_PER_CLUSTER = np.zeros([self.window_size, SCTA.MAX_CLUSTER_ID])

        # Contains cluster formed after each timestamp along with the central node
        reserved_central_node_list = []
        reserved_central_node_communities_size = []
        reserved_central_node_communities = []


        for ep_idx in range(self.window_size):
            if ep_idx%100 == 0:
                print(ep_idx)

            node_community = {} # {node: cluster_id}

            # Filter central nodes (1)
            filtered_central_nodes_having_max_recehability = []

            for idx, nodes in enumerate(reserved_central_node_list):

                """
                1. Out of five possible central nodes idenitfy just one central node from where to 
                   start traversal. This corresponds to the one which results in biggest cluster 
                   size during trial run. Choosing one central node make the handling weird 
                   scenarios of cluster merging much easier.

                2. Step one will give distinct central nodes corresponding to distinct clusters
                   (or atleast distinct clusters from previous epochs)

                3. Proceede as usual. If two central nodes merge previously written code will 
                   handle it!
                """

                central_nodes_reachability = []

                for n_idx, node in enumerate(nodes):

                    node_community = {}
                    node_community[node] = reserved_central_node_communities[idx]

                    node_community = self.expand_cluster(self.bin_conn, node, 
                        reserved_central_node_communities[idx], 
                        node_community, 
                        reserved_central_node_communities, 
                        reserved_central_node_communities_size, 
                        self.local_order[ep_idx, :],
                        self.sync_thrsh)

                    # Number of nodes which can be reached via current central node
                    count = len(node_community)

                    central_nodes_reachability.append(count)

                filtered_central_node = nodes[central_nodes_reachability.index(
                    max(central_nodes_reachability))]
                filtered_central_nodes_having_max_recehability.append(filtered_central_node)

            # We have the filtered central nodes now! (2)
            # Initialize central node list and corresponding cluster id ( from previous timestamp )
            for idx, node in enumerate(filtered_central_nodes_having_max_recehability):
                node_community[node] = reserved_central_node_communities[idx]

            # 3. Expand communities around central nodes (in order of the central node list)
            for idx, node in enumerate(filtered_central_nodes_having_max_recehability):
                # before bfs check if this central node 
                # already became part of some other central node
                node_community = self.expand_cluster(self.bin_conn, node, 
                    reserved_central_node_communities[idx], 
                    node_community, 
                    reserved_central_node_communities, 
                    reserved_central_node_communities_size, 
                    self.local_order[ep_idx, :],
                    self.sync_thrsh)

            # Expand communities for the ramining nodes (4)
            for i in range(self.N):
                """
                Iterate each node and perform bfs only if its local order is 
                greater than some threshold, # and it is not part of any community it.

                If the node is part of some community, this suggest that it has already
                reached out to neigbours and inducted them into its community.

                Each node should be given a chance to become a starting node in order to 
                make sure all the edges are visited.
                """
                if self.local_order[ep_idx, i] > thr: 

                    if ( i not in node_community.keys() ):
                        # This node has no comunity.
                        # Generate some unused community id
                        unique_values = set(node_community.values())

                        community_id = SCTA.MIN_CLUSTER_ID
                        if unique_values:
                            community_id = max(unique_values) + 1

                        node_community[i] = community_id

                    # Now find all the cluster which belongs to this community
                    node_community = self.expand_cluster(self.bin_conn, 
                        i, 
                        node_community[i], 
                        node_community, 
                        reserved_central_node_communities, 
                        reserved_central_node_communities_size, 
                        self.local_order[ep_idx, :],
                        self.sync_thrsh)

            # Idenitify communities and corresponding central nodes (5)
            reserved_central_node_list = []                # cleanup from previous run
            reserved_central_node_communities = []         # cleanup from previous run
            reserved_central_node_communities_size = []    # cleanup from previous run
            communities = {}                               # {community_id: [node list]}

            for node, community_id in node_community.items():
                # value: community id, key: node index
                if community_id in communities:
                    communities[community_id].append(node)
                else:
                    communities[community_id] = []
                    communities[community_id].append(node)

            for community_id in communities.keys():
                # Minimum 2 nodes should be present to be considered a cluster
                if len(communities[community_id]) > 2:

                    # Pick the node having highest local order
                    indexes = communities[community_id]
                    central_nodes = []
                    if len(communities[community_id]) > 10:
                        """
                        Scenario: A central node which was part of biggest cluster at t=0
                        detaches from main sync. cluster at t=1. However, at t=1 this central 
                        node is expanded to form form cluster, however, resulting in cluster
                        of cluster of size 1. Now, the index of biggest cluster changes becuase
                        this central node has eaten up the index corresponding to biggest 
                        cluster.

                        This can be avoided by choosing multiple central nodes and 
                        expand in next iteration in hope that atleast one of them 
                        has not detached form main synchronization cluster. If this is
                        true then the biggest cluster will have the same index across
                        epochs. However, there will be few central nodes which are not
                        part of main sync. cluster anymore but have the cluster id same
                        as the main sync cluster. But thats okay! They will die out in 
                        next epoch if still detached.
                        """
                        central_nodes = self.get_central_node(ep_idx, indexes, 5)
                    else:
                        central_nodes = self.get_central_node(ep_idx, indexes, 1)
                    
                    central_nodes_community = [community_id] * len(central_nodes)
                    central_nodes_size = [len(communities[community_id])]*len(central_nodes)

                    reserved_central_node_list.append(central_nodes)
                    reserved_central_node_communities.append(community_id)
                    reserved_central_node_communities_size.append(len(communities[community_id]))

            # Sort central node list according to community size (6)
            sorted_indexes = np.argsort(reserved_central_node_communities_size)[::-1]

            reserved_central_node_communities_size = [reserved_central_node_communities_size[i] \
                                                      for i in sorted_indexes]
            reserved_central_node_list = [reserved_central_node_list[i] for i in sorted_indexes]
            reserved_central_node_communities = [reserved_central_node_communities[i] \
                                                 for i in sorted_indexes]

            # Keep track of communities at the end of each epoch (7)
            for node, community_id in node_community.items():
                NODE_COMMUNITIES[ep_idx, node] = community_id

            # Update time of entry in the latest current cluster (8)
            # (current latest cluster is the biggest cluster for this node)
            
            for i in range(self.N):
                node_current_membership = 0 if (i not in node_community.keys()) \
                    else node_community[i]
                
                # Node community has changed! Update its time of entry
                # For now, Don't worry about node getting detached from the community!
                if NODE_COMMUNITIES[ep_idx-1, i] != NODE_COMMUNITIES[ep_idx, i]:
                    TIME_OF_ENTRY[i] = ep_idx
                elif node_current_membership == 0: 
                    # Keep updating its time of entry because it has not
                    # entered the main sync cluster.
                    TIME_OF_ENTRY[i] = ep_idx
        
        self.node_communities[:] = NODE_COMMUNITIES
        print("Cluster tracking complete!")
    
    def change_membership(self, 
                          node_community, 
                          from_community, 
                          to_community):
        # This modifies the original dictionary
        for node, community_id in node_community.items():
            if community_id == from_community:
                node_community[node] = to_community
        
        return node_community

    def expand_cluster(self, 
                       adj_matrix, 
                       start_node, 
                       start_community_id, 
                       node_community, 
                       central_node_communities,
                       central_node_communities_size, 
                       local_order_list, 
                       thr):

        """
        Each function call expands just one community i.e. 'start_community_id'
        """
        num_nodes = len(adj_matrix)
        visited = set()
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            visited.add(node)
            
            # Enqueue the unvisited adjacent nodes
            for neighbor in range(num_nodes):
                if (adj_matrix[node, neighbor] == 1) \
                    and (neighbor not in visited) \
                    and (local_order_list[neighbor] > thr):
                    """
                    This node is being visited first time and is not part of any cluster.
                    Since, the local order for this node is above some threshold include it 
                    in the community defined by central node.

                    No need to add this node to queue here. Nevertheless, 
                    this node will be given a chance to be a starting node. 
                    So, we are good!
                    """
                    # queue.append(neighbor) # Adding to queue here terribly slows down the system
                    # 'neighbour' already part of some community
                    if neighbor in node_community.keys():

                        """
                        What to do if this node already has some communtiy id assigned
                        Three scenarios:
                        1. Both nodes belong to community defined by one of central nodes
                        2. one of the nodes belong to community defined by one of central nodes
                        3. none of the nodes community belong to central nodes' community
                        """

                        # 'neighbor' node has different community
                        # This scenraio occurs because of non-symmetry of directed graph
                        # and might occur very rarely for mouse brain network. Wait to see!
                        if node_community[node] != node_community[neighbor]:

                            # What should be the merged community id?
                            merged_community_id = node_community[node]

                            if (node_community[node] in central_node_communities) \
                                and (node_community[neighbor] in central_node_communities):
                                
                                # Both communites are extension of communities formed by 
                                # central nodes i.e. two clusters expanded around central 
                                # node. Its time to merge these two different cluster 
                                # into one big cluster.

                                idx_node = central_node_communities.index(node_community[node])
                                idx_neighbor = central_node_communities.index(
                                    node_community[neighbor])

                                # Which community was larger in previous epoch?
                                if central_node_communities_size[idx_node] > \
                                    central_node_communities_size[idx_neighbor]:
                                    node_community = self.change_membership(node_community, 
                                                      node_community[neighbor], 
                                                      node_community[node])
                                else:
                                    node_community = self.change_membership(node_community, 
                                                      node_community[node], 
                                                      node_community[neighbor])

                            elif node_community[node] in central_node_communities:
                                
                                # Merge the node with central node cluster.
                                node_community = self.change_membership(node_community, 
                                                  node_community[neighbor], 
                                                  node_community[node])

                            elif node_community[neighbor] in central_node_communities:
                                
                                # Merge the node with central node cluster.
                                node_community = self.change_membership(node_community, 
                                                  node_community[node], 
                                                  node_community[neighbor])

                            else:
                                # None of the nodes belong to community formed by central node. 
                                # Merge to a lower community id
                                
                                if node_community[node] < node_community[neighbor]:
                                    node_community = self.change_membership(node_community, 
                                                      node_community[neighbor], 
                                                      node_community[node])
                                else:
                                    node_community = self.change_membership(node_community, 
                                                      node_community[node], 
                                                      node_community[neighbor])
                        else:
                            # Both already in same communtiy. Do nothing!
                            xwdsf = 5 
                    else:
                        if node not in node_community.keys():
                            print("Problem")

                        # This neighbor is visited for the first time so 
                        # append it to the queue
                        # This speeds up the process
                        queue.append(neighbor) 
                        node_community[neighbor] = node_community[node]
        
        return node_community

    def update_synchronization_cluster_stats(self):
        for t in range(self.window_size):
            unique_values, value_counts = np.unique(self.node_communities[t,:], 
                                                    return_counts=True)
            self.count_per_sync_cluster[t, unique_values.astype(int)] = value_counts