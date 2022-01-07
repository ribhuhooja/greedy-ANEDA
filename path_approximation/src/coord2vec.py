from os import spawnlp
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from data_helper import write_file


# https://github.com/dmlc/dgl/tree/e667545da55017d5dbbd3f243d986506284d3e41/examples/pytorch/node2vec
class Coord2vec(nn.Module):
    """Node2vec model from paper path_approximation: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(self, dataset, num_nodes, embedding_dim, init_embeddings=None, use_sparse=True):
        super(Coord2vec, self).__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.N = num_nodes
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')
        
        # Add an additional embedding for the -1 case in the random walk 
        # (walk ended early because it reached a node with no outward edges)
        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)
        if init_embeddings is not None:
            self.embedding.weight = nn.Parameter(init_embeddings)


        # self.register_parameter(name="bias", param=torch.nn.Parameter(dataset[:, 2].mean()))
        self.register_parameter(name="bias", param=torch.nn.Parameter(torch.tensor(0.0)))

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding
        """

        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, samples):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
        neg_trace: Tensor
            negative random walk trace
        """

        # Positive
        left, right, dist = samples[:, 0].long(), samples[:, 1].long(), samples[:, 2]

        left_emb = self.embedding(left)
        right_emb = self.embedding(right)
        out = (left_emb * right_emb).sum(dim=-1).view(-1)
        # print(out.mean(), dist.mean())

        # 1 -> 0
        # -1 -> inf

        # compute loss
        h = torch.exp(-out)
        # h = torch.exp(self.bias-out)
        # print("{0:.3f}, {1:.3f}, {2:.3f}".format(h.mean().item(), torch.median(h).item(), dist.mean().item()))
        # print("{0:.3f}".format(h.mean().item()), end=" ")
        loss = self.loss_fn(h, dist)

        return loss

    def loader(self, batch_size):
        """
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        DataLoader
            Coord2vec training data loader
        """
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    @torch.no_grad()
    def evaluate(self, x_train, y_train, x_val, y_val):
        """
        Evaluate the quality of embedding vector via a downstream classification task with logistic regression.
        """
        x_train = self.forward(x_train)
        x_val = self.forward(x_val)

        x_train, y_train = x_train.cpu().numpy(), y_train.cpu().numpy()
        x_val, y_val = x_val.cpu().numpy(), y_val.cpu().numpy()
        lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150).fit(x_train, y_train)

        return lr.score(x_val, y_val)


class Coord2vecModel(object):
    """
    Wrapper of the ``Coord2Vec`` class with a ``train`` method.
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, uses PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly. Default: ``None``.
    eval_set: list of tuples (Tensor, Tensor)
        [(nodes_train,y_train),(nodes_val,y_val)]
        If omitted, model will not be evaluated. Default: ``None``.
    eval_steps: int
        Interval steps of evaluation.
        if set <= 0, model will not be evaluated. Default: ``None``.
    device: str
        device, {'cpu', 'cuda'}, default 'cpu'
    """

    def __init__(self, dataset, num_nodes, embedding_dim, init_embeddings=None, use_sparse=True, eval_set=None, eval_steps=-1, device='cpu'):

        self.model = Coord2vec(dataset, num_nodes, embedding_dim, init_embeddings, use_sparse)
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.eval_set = eval_set

        if device == 'cpu':
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"...using {self.device}")

    def _train_step(self, model, loader, optimizer, device, bias_optimizer=None):
        model.train()
        total_loss = 0
        for samples in loader:
            samples = samples.to(device)
            optimizer.zero_grad()
            if bias_optimizer:
                bias_optimizer.zero_grad()
            loss = model.loss(samples)
            loss.backward()
            optimizer.step()
            if bias_optimizer:
                bias_optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate_step(self):
        nodes_train, y_train = self.eval_set[0]
        nodes_val, y_val = self.eval_set[1]

        acc = self.model.evaluate(nodes_train, y_train, nodes_val, y_val)
        return acc

    def train(self, epochs, batch_size, learning_rate=0.01):
        """
        Parameters
        ----------
        epochs: int
            num of train epoch
        batch_size: int
            batch size
        learning_rate: float
            learning rate. Default 0.01.
        """

        self.model = self.model.to(self.device)
        loader = self.model.loader(batch_size)
        if self.use_sparse:
            params = list(self.model.parameters())
            optimizer = torch.optim.SparseAdam(params[1:], lr=learning_rate)
            bias_optimizer = torch.optim.Adam(params[0:1], lr=learning_rate*0.1)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            bias_optimizer = None

        for i in range(epochs):
            # print("Bias: {0:.3f}".format(self.model.bias.item()))
            loss = self._train_step(self.model, loader, optimizer, self.device, bias_optimizer)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    if self.eval_set is not None:
                        acc = self._evaluate_step()
                        print("Epoch: {}, Train Loss: {:.4f}, Val Acc: {:.4f}".format(i + 1, loss, acc))
                    else:
                        print("Epoch: {}, Train Loss: {:.4f}".format(i + 1, loss))
            print()

    def embedding(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding.
        """

        return self.model(nodes)


def run_coord2vec(dataset, num_nodes, init_embeddings=None, eval_set=None, args=None, output_path=None):
    t_tick = time.time()  ## start measuring running time
    if args is None:
        raise ValueError("need args for coord2vec!")

    # Convert dict to tuple
    Coord2VecParams = namedtuple('Coord2VecParams', args)
    args = Coord2VecParams(**args)

    # train Coord2Vec
    print("training Coord2Vec, it will take a while...")
    print("coord2vec's arguments:", args)
    trainer = Coord2vecModel(dataset,
                            num_nodes=num_nodes,
                            embedding_dim=args.embedding_dim,
                            init_embeddings=init_embeddings,
                            eval_set=eval_set,
                            eval_steps=1,
                            device=args.device)

    trainer.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

    t_tock = time.time()

    print(f"done training coord2vec, running time = {np.round((t_tock - t_tick) / 60)} minutes.")
    # Calc embedding
    embedding = trainer.embedding().data

    if args.device == "cuda":
        embedding = embedding.cpu().numpy()

    write_file(output_path, embedding)
    return embedding