import .GromovWassersteinGraphToolkit as GwGt
import pickle
import numpy as np
from scipy.sparse import csr_matrix

def adjacence_matrix_from_edge_index(edge_index, v):
    '''
        graph(undirected or directed)

        parameters:
        edge_index: shape 2 * N
        v: number of vertices

        return:
        pairwise confidence of matching or similarity
    '''
    ret = np.zeros(v, v, type = np.float32)
    for i in range(edge_index.shape[1]):
        ret[edge_index[0, i], edge_index[1, i]] = 1.0
    return csr_matrix(ret)


def compute_similarity(edge_index_1, prior_1, edge_index_2, prior_2):
    num_iter = 2000
    ot_dict = { 'loss_type': 'L2',  # the key hyperparameters of GW distance
                'ot_method': 'proximal',
                'beta': 0.025,
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
    adjacence_1 = adjacence_matrix_from_edge_index(edge_index_1)
    adjacence_2 = adjacence_matrix_from_edge_index(edge_index_2)
    idx2node_1 = { i: str(i) for i in range(prior_1) }
    idx2node_2 = { i: str(i) for i in range(prior_2) }
    pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
    adjacence_1, adjacence_2, prior_1, prior_2, idx2node_1, idx2node_2, ot_dict,
    weights=None, predefine_barycenter=False, cluster_num=2,
    partition_level=3, max_node_num=0)
    return pairs_confidence
