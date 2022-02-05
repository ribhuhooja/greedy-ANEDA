from os import spawnlp
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from data_helper import write_file
from evaluations import evaluate_metrics

# https://github.com/dmlc/dgl/tree/e667545da55017d5dbbd3f243d986506284d3e41/examples/pytorch/node2vec
class CollaborativeFiltering(nn.Module):
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

    def __init__(self, dataset, num_nodes, embedding_dim, init_embeddings=None, loss_func="mse", measure="norm", norm=2, use_sparse=True, device=None):
        super(CollaborativeFiltering, self).__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.N = num_nodes

        assert loss_func in ["mse", "mre", "poisson"]
        if loss_func == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_func == "mre":
            eps = torch.tensor(1e-10, device=device)
            self.loss_fn = lambda pred, true : (torch.abs(pred - true) / torch.maximum(torch.abs(true), eps)).mean()
        elif loss_func == "poisson":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')
        
        assert measure in ["norm", "hyperbolic", "spherical", "lat-long", "inv-dot"]
        self.measure = measure
        self.norm = norm
        # self.loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')
        # print(self.loss_fn)
        # max_loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='none')
        # self.max_loss_fn = lambda x, y: torch.mean(torch.topk(y-x, len(x)//100)[0])/100
        
        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)
        print(torch.mean(torch.linalg.norm(init_embeddings, dim=1)))
        if init_embeddings is not None:
            self.embedding.weight = nn.Parameter(init_embeddings)        
        self.embedding = self.embedding.to(device)
        self.device = device
        print(self.embedding.weight.device)

        self.ones = torch.ones(self.N, 1).to(device)
        
        # def gradInspect(self, input, output):
        #     # grad = input[0].coalesce()
        #     # print(grad)

        #     # grad = output[0]
        #     # print(input[0].size(), output[0].size())
        #     # print(input)
        #     # print(output)
        #     # print(torch.isnan(output[0]).nonzero())
        #     # grad = grad.to_dense()
        #     # print(torch.isnan(grad).nonzero()[:10])
        #     # print(output[0])
        #     print("backward", torch.sum(torch.isinf(output[0])).item(), torch.sum(torch.isnan(output[0])).item())

        #     # print()
        # self.embedding.register_backward_hook(gradInspect)

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

    def loss(self, samples, return_pred=False):
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

        if self.measure == "norm":
            # left_emb = left_emb.unsqueeze(dim=1)
            # right_emb = right_emb.unsqueeze(dim=1)
            dist_hat = torch.linalg.norm(left_emb-right_emb, ord=self.norm, dim=1) # torch.cdist(left_emb, right_emb, p=self.norm).view(-1)
        elif self.measure == "hyperbolic":
            R = 1 # 6731
            # left_right_norm = torch.linalg.norm(left_emb-right_emb, dim=1) # torch.cdist(left_emb, right_emb).view(-1)
            # left_norm = torch.linalg.norm(left_emb, dim=1) # torch.cdist(left_emb, left_emb).view(-1)
            # right_norm = torch.linalg.norm(right_emb, dim=1) # torch.cdist(right_emb, right_emb).view(-1)
            # delta = torch.div(left_right_norm**2, (1-left_norm**2)*(1-right_norm**2))
            # dist_hat = R*torch.arccosh(1+2*delta)

            # errs = torch.isnan(dist_hat).nonzero(as_tuple=True)[0]
            # if len(errs) > 0:
            #     print("tests")
            #     print(left_norm[0], left_norm[errs[:10]])
            #     print(right_norm[0], right_norm[errs[:10]])
            #     print(left_right_norm[0], left_right_norm[errs[:10]])
            #     print(delta[0], delta[errs[:10]])
            #     print(dist_hat[0], dist_hat[errs[:10]])
            #     print()
            ones = torch.ones(len(left_emb)).to(self.device)
            left_bias = (ones + torch.linalg.norm(left_emb, dim=1)**2)**1/2
            right_bias = (ones + torch.linalg.norm(right_emb, dim=1)**2)**1/2
            mink_prod = left_bias*right_bias - torch.sum(left_emb*right_emb, dim=1)
            dist_hat = torch.arccosh(torch.maximum(mink_prod, ones))
            # print(torch.mean(left_bias))
            # print(torch.mean(mink_prod))
            # print()
            # one = torch.tensor(1)
            # dist_hat = torch.arccosh(torch.max(mink_prod, one))
            
            # errs = torch.isnan(dist_hat).nonzero(as_tuple=True)[0]
            # if len(errs) > 0:
            #     print("tests")
            #     print(left_norm[0], left_norm[errs[:10]])
            #     print(right_norm[0], right_norm[errs[:10]])
            #     print(left_right_norm[0], left_right_norm[errs[:10]])
            #     print(delta[0], delta[errs[:10]])
            #     print(dist_hat[0], dist_hat[errs[:10]])
            #     print()
        elif self.measure == "spherical":
            R = 1 # 6731
            dot = torch.sum(left_emb*right_emb, dim=1)
            left_norm = torch.linalg.norm(left_emb, dim=1)
            right_norm = torch.linalg.norm(right_emb, dim=1)
            # Clamp values to prevent floating point error, which results in nan after arccos
            div = torch.clamp(torch.div(dot, left_norm * right_norm), min=-1, max=1)
            dist_hat = R*torch.arccos(div) 
        elif self.measure == "lat-long":
            R = 1 # 6731
            delta = right_emb - left_emb
            d = 0.5 - torch.cos(delta[:, 0])/2 + torch.cos(left_emb[:, 0])*torch.cos(right_emb[:, 0]) * (1-torch.cos(delta[:, 1]))/2
            dist_hat = 2*R*torch.arcsin(torch.sqrt(d))
        elif self.measure == "inv-dot":
            dot = torch.sum(left_emb*right_emb, dim=1)
            dist_hat = torch.exp(-dot)

        # if self.use_hyperbolic:
        #     R = 6731
        #     left_norm = torch.cdist(left_emb[:,:,:3], left_emb[:,:,:3]).view(-1) # torch.linalg.norm(left_emb[:, :3], dim=1)
        #     right_norm = torch.cdist(right_emb[:,:,:3], right_emb[:,:,:3]).view(-1) # torch.linalg.norm(right_emb[:, :3], dim=1) 
        #     left_right_norm = torch.cdist(left_emb[:,:,:3], right_emb[:,:,:3]).view(-1) # torch.linalg.norm(left_emb[:, :3] - right_emb[:, :3], dim=1)
        #     delta = torch.div(left_right_norm**2, (1-left_norm**2)*(1-right_norm**2))
        #     # print(left_emb.size(), left_norm.size(), right_norm.size(), left_right_norm.size(), delta.size())
        #     h = R*torch.arccosh(1+2*delta) + torch.cdist(left_emb[:,:,3:], right_emb[:,:,3:]).view(-1)
        # else:
        #     h = torch.cdist(left_emb, right_emb).view(-1)
        loss = self.loss_fn(dist_hat, dist) # + self.max_loss_fn(h, dist)
        # if self.measure == "hyperbolic":
        #     loss = loss + torch.linalg.norm(1 - right_emb[:,0]**2 + torch.sum(right_emb[:,1:]**2, axis=1))
        # print("forward", torch.sum(torch.isnan(dist_hat)).item())
        # print()

        if return_pred:
            return loss, dist, dist_hat
        return loss

    def loader(self, dataset, batch_size):
        """
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        DataLoader
            Collaborative Filtering training data loader
        """
        if batch_size == 0:
            batch_size = len(dataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


class CollaborativeFilteringModel(object):
    """
    Wrapper of the ``CollaborativeFiltering`` class with a ``train`` method.
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

    def __init__(self, dataset, num_nodes, embedding_dim, init_embeddings=None, loss_func="mse", measure="norm", norm=2, use_sparse=True, eval_set=None, eval_steps=-1, device='cpu'):
        if device == 'cpu':
            self.device = device
        else:
            # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)

        print(f"...using {self.device}")

        self.model = CollaborativeFiltering(dataset, num_nodes, embedding_dim, init_embeddings, loss_func, measure, norm, use_sparse, device)
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.dataset = dataset
        self.eval_set = eval_set

    def _train_step(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for samples in loader:
            samples = samples.to(device)
            optimizer.zero_grad()
            loss = model.loss(samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate_step(self, val_loader, evaluate_function=None, config=None):
        yhat_list = []
        ytrue_list = []
        val_losses = []
        metric_scores = None
        with torch.no_grad():
            for samples in val_loader:
                samples = samples.to(self.device)
                val_loss, y_val, yhat = self.model.loss(samples, return_pred=True)
                val_losses.append(val_loss.item())
                y_val, yhat = y_val.view(-1, 1), yhat.view(-1, 1)

                if evaluate_function:
                    if self.device != "cpu":
                        yhat_list.append(yhat.cpu().numpy())
                        ytrue_list.append(y_val.cpu().numpy())
                    else:
                        yhat_list.append(yhat.numpy())
                        ytrue_list.append(y_val.numpy())
            print((yhat*1.0).mean().item(), (y_val*1.0).mean().item())
            validation_loss = np.mean(val_losses)
        if evaluate_function:
            metric_scores = evaluate_function(np.vstack(ytrue_list).reshape(-1), np.vstack(yhat_list).reshape(-1),
                                              print_out=False, config=config)
        return validation_loss, metric_scores


    def train(self, epochs, batch_size, learning_rate=0.01, optimizer="adam", config=None):
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
        loader = self.model.loader(self.dataset, batch_size)
        val_loader = self.model.loader(self.eval_set, batch_size)
        
        assert optimizer in ["adam", "sgd"]
        if optimizer == "adam":
            if self.use_sparse:
                optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        for i in range(epochs):
            # print("Bias: {0:.3f}".format(self.model.bias.item()))
            loss = self._train_step(self.model, loader, optimizer, self.device)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    if self.eval_set is not None:
                        validation_loss, val_metrics = self._evaluate_step(val_loader, evaluate_function=evaluate_metrics, config=config)
                        print("Epoch: {}, Train Loss: {:.4f};\tValidation loss: {:.4f}, Validation metrics: {}".format(i + 1, loss, validation_loss, val_metrics))
                    else:
                        print("Epoch: {}, Train Loss: {:.4f}".format(i + 1, loss))
            # exit()
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


def run_collab_filtering(dataset, num_nodes, init_embeddings=None, eval_set=None, args=None, output_path=None, config=None):
    t_tick = time.time()  ## start measuring running time
    if args is None:
        raise ValueError("need args for collaborative filtering!")

    # Convert dict to tuple
    CollabFilteringParams = namedtuple('CollabFilteringParams', args)
    args = CollabFilteringParams(**args)

    # train CollaborativeFiltering
    print("training CollaborativeFiltering, it will take a while...")
    print("CollaborativeFiltering arguments:", args)
    trainer = CollaborativeFilteringModel(dataset,
                            num_nodes=num_nodes,
                            embedding_dim=args.embedding_dim,
                            init_embeddings=init_embeddings,
                            loss_func=args.loss_func,
                            measure=args.measure,
                            norm=args.norm,
                            eval_set=eval_set,
                            eval_steps=1,
                            device=args.device)

    trainer.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, optimizer=args.optimizer, config=config)

    t_tock = time.time()

    print(f"done training CollaborativeFiltering, running time = {np.round((t_tock - t_tick) / 60)} minutes.")
    # Calc embedding

    embedding = trainer.embedding().data

    if "cuda" in args.device:
        embedding = embedding.cpu().numpy()

    write_file(output_path, embedding)
    return embedding