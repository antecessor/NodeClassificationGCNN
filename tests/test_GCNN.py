from unittest import TestCase

from GCNN import NodeClassificationGCNN
import torch


class TestNodeClassificationGCNN(TestCase):
    def setUp(self) -> None:
        super().setUp()
        node_numbers = 5
        hidden_dim = 10
        feature_dim = 7
        nclass = 2

        self.nodeClassificationGCNN = NodeClassificationGCNN(feature_dim, hidden_dim, nclass)

        self.nodesRepresentation = torch.nn.Parameter(torch.rand(node_numbers, feature_dim))
        connections = torch.tensor([[0, 0, 1, 4, 3, 2], [1, 2, 3, 2, 1, 3]])
        self.adj = torch.sparse.FloatTensor(connections, torch.ones(connections.shape[1]), torch.Size([node_numbers, node_numbers])).to_dense()
        self.adj_plusEye = self.adj + torch.eye(node_numbers, node_numbers)
        self.d = torch.diag(self.adj_plusEye.sum(dim=1))
        d_inverse = torch.inverse(torch.sqrt(self.d))
        self.normalized_adj = d_inverse * self.adj_plusEye * d_inverse

    def test_NodeClassificationGCNN(self):
        out = self.nodeClassificationGCNN.forward(self.nodesRepresentation, self.normalized_adj)
        pass
