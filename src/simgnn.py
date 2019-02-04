import glob
import time
import math
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged

class SimGNN(torch.nn.Module):

    def __init__(self, args, number_of_labels):
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons 



    def setup_layers(self):
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1,1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1,-1)
        return hist
      
         


    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, training=self.training)
        features = self.convolution_3(features, edge_index)
        return features


    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        if self.args.histogram == True:
            hist =self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))
        
        
        pooled_features_1 =  self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        if self.args.histogram == True:
            scores = torch.cat((scores,hist),dim=1).view(1,-1)
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))

        return score


class SimGNNTrainer(object):

    def __init__(self, args):
        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        self.model = SimGNN(self.args, self.number_of_labels)


    def initial_label_enumeration(self):
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
         

    def create_batches(self):
        random.shuffle(self.training_graphs)
        batches = [self.training_graphs[graph:graph+self.args.batch_size] for graph in range(0, len(self.training_graphs), self.args.batch_size)]
        return batches

    def transfer_to_torch(self, data):
        new_data = dict()
        edges_1 = torch.from_numpy(np.array(data["graph_1"], dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(data["graph_2"], dtype=np.int64).T).type(torch.long)

        features_1 = torch.FloatTensor(np.array([[ 1.0 if int(self.global_labels[node]) == int(label) else 0 for label in self.global_labels] for node in data["labels_1"]]))
        features_2 = torch.FloatTensor(np.array([[ 1.0 if int(self.global_labels[node]) == int(label) else 0 for label in self.global_labels] for node in data["labels_2"]]))
        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        new_data["features_2"] = features_2
        normalized_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        new_data["target"] =  torch.from_numpy(np.exp(-normalized_ged).reshape(1,1)).view(-1).float()

        return new_data

    def process_batch(self,batch):
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            target = self.model(data)
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(data["target"],prediction)
        losses.backward(retain_graph = True)
        self.optimizer.step()
        return losses.item()


    def fit(self):
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                loss_score = self.process_batch(batch)
                index = index + len(batch)
                self.loss_sum = self.loss_sum + loss_score
            loss = self.loss_sum/index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

    def score(self):
        print("\n\nModel evaluation.\n")
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)
            self.ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = self.model(data)
            prediction = self.model(data)
            self.scores.append(calculate_loss(prediction, target))
        self.print_evaluation()

    def print_evaluation(self):
        norm_ged_mean = np.mean(self.ground_truth)
        base_error= np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error,5))+".")
        print("\nModel test error: " +str(round(model_error,5))+".")
        
