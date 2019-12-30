from . import GromovWassersteinGraphToolkit as GwGt
import pickle
import numpy as np
from scipy.sparse import csr_matrix

def adjacency_matrix_from_edge_index(edge_index, v):
    '''
        graph(undirected or directed)

        parameters:
        edge_index: shape 2 * N
        v: number of vertices

        return:
        pairwise confidence of matching or similarity
    '''
    ret = np.zeros((v, v), np.float32)
    for i in range(edge_index.shape[1]):
        ret[edge_index[0, i], edge_index[1, i]] = 1.0
    return csr_matrix(ret)


def compute_similarity(edge_index_1, prior_1, edge_index_2, prior_2):
    num_iter = 2000
    ot_dict = { 'loss_type': 'L2',  # the key hyperparameters of GW distance
                'ot_method': 'proximal',
                'beta': 0.15,
                'outer_iteration': num_iter,
                # outer, inner iteration, error bound of optimal transport
                'iter_bound': 1e-30,
                'inner_iteration': 2,
                'sk_bound': 1e-30,
                'node_prior': 1e3,
                'max_iter': 4,      # iteration and error bound for calcuating barycenter
                'cost_bound': 1e-26,
                'update_p': False,  # optional updates of source distribution
                'lr': 0,
                'alpha': 0 }
    print(prior_1, prior_2)
    v1, v2 = prior_1.shape[0], prior_2.shape[0]
    v = max(v1, v2)
    adjacency_1 = adjacency_matrix_from_edge_index(edge_index_1, v1)
    adjacency_2 = adjacency_matrix_from_edge_index(edge_index_2, v2)
    idx2node_1 = { i: str(i) for i in range(prior_1.shape[0]) }
    idx2node_2 = { i: str(i) for i in range(prior_2.shape[0]) }
    # pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
    # adjacency_1, adjacency_2, prior_1, prior_2, idx2node_1, idx2node_2, ot_dict,
    # weights=None, predefine_barycenter=False, cluster_num=2,
    # partition_level=3, max_node_num=0)
    swap = v1 < v2
    # default ns >= nt
    if not swap:
        # pairs_idx, pairs_name, pairs_confidence = GwGt.direct_graph_matching(
        #    adjacency_1, adjacency_2, prior_1, prior_2, idx2node_1, idx2node_2, ot_dict)
        pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
        adjacency_1, adjacency_2, prior_1, prior_2, idx2node_1, idx2node_2, ot_dict,
        weights=None, predefine_barycenter=False, cluster_num=2, partition_level=3, max_node_num=0)
    else:
        # pairs_idx, pairs_name, pairs_confidence = GwGt.direct_graph_matching(
        #     adjacency_2, adjacency_1, prior_2, prior_1, idx2node_2, idx2node_1, ot_dict)
        pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
        adjacency_2, adjacency_1, prior_2, prior_1, idx2node_2, idx2node_1, ot_dict,
        weights=None, predefine_barycenter=False, cluster_num=2, partition_level=3, max_node_num=0)
    ret = np.zeros((v, v), np.float32)
    # source, target also means graph1 and graph2
    # print(pairs_confidence)
    for i, (si, ti) in enumerate(pairs_idx):
        if not swap:
            ret[si, ti] = pairs_confidence[i]
        else:
            ret[ti, si] = pairs_confidence[i]
    return ret
