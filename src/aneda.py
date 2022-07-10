import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_helper import write_file
from evaluations import evaluate_metrics

def get_distance(left_emb, right_emb, measure, norm=2, graph_diameter=1, device="cpu"):
    if measure == "norm":
        dist_hat = torch.linalg.norm(left_emb-right_emb, ord=norm, dim=1)
    elif measure == "poincare":
        eps = 1e-5
        left_right_norm = torch.linalg.norm(left_emb-right_emb, dim=1)
        left_norm = torch.linalg.norm(left_emb, dim=1)
        right_norm = torch.linalg.norm(right_emb, dim=1)
        delta = torch.clamp(torch.div(left_right_norm**2, (1-left_norm**2 + eps)*(1-right_norm**2 + eps)), min=0)
        dist_hat = torch.arccosh(1+2*delta)
        # print(torch.mean(dist_hat).item(), torch.mean(left_right_norm).item(), torch.mean(left_norm).item(), torch.mean(right_norm).item(), torch.sum(torch.isnan(dist_hat)).item())
    elif measure == "hyperboloid":
        ones = torch.ones(len(left_emb)).to(device)
        left_bias = (ones + torch.linalg.norm(left_emb, dim=1)**2)**1/2
        right_bias = (ones + torch.linalg.norm(right_emb, dim=1)**2)**1/2
        mink_prod = left_bias*right_bias - torch.sum(left_emb*right_emb, dim=1)
        dist_hat = torch.arccosh(torch.maximum(mink_prod, ones))
    elif measure == "spherical":
        dot = torch.sum(left_emb*right_emb, dim=1)
        left_norm = torch.linalg.norm(left_emb, dim=1)
        right_norm = torch.linalg.norm(right_emb, dim=1)
        # Clamp values to prevent floating point error, which results in nan after arccos
        div = torch.clamp(torch.div(dot, left_norm * right_norm), min=-1, max=1)
        dist_hat = torch.arccos(div) *(graph_diameter / (np.pi/2))
    elif measure == "lat-long":
        delta = right_emb - left_emb
        d = 0.5 - torch.cos(delta[:, 0])/2 + torch.cos(left_emb[:, 0])*torch.cos(right_emb[:, 0]) * (1-torch.cos(delta[:, 1]))/2
        dist_hat = 2*torch.arcsin(torch.sqrt(d))
    elif measure == "inv-dot":
        dot = torch.sum(left_emb*right_emb, dim=1)
        left_norm = torch.linalg.norm(left_emb, dim=1)
        right_norm = torch.linalg.norm(right_emb, dim=1)
        div = torch.clamp(torch.div(dot, left_norm * right_norm), min=-1, max=1)
        dist_hat = -(div-1)*graph_diameter/2
    return dist_hat

def get_distance_numpy(left_emb, right_emb, measure, norm=2, graph_diameter=1):
    if measure == "norm":
        dist_hat = np.linalg.norm(left_emb-right_emb, ord=norm)
    elif measure == "poincare":
        left_right_norm = np.linalg.norm(left_emb-right_emb)
        left_norm = np.linalg.norm(left_emb)
        right_norm = np.linalg.norm(right_emb)
        delta = np.divide(left_right_norm**2, (1-left_norm**2)*(1-right_norm**2))
        dist_hat = np.arccosh(1+2*delta)
    elif measure == "hyperboloid":
        ones = np.ones(len(left_emb))
        left_bias = (ones + np.linalg.norm(left_emb)**2)**(1/2)
        right_bias = (ones + np.linalg.norm(right_emb)**2)**(1/2)
        mink_prod = left_bias*right_bias - np.dot(left_emb, right_emb)
        dist_hat = np.arccosh(np.maximum(mink_prod, ones))
    elif measure == "spherical":
        dot = np.dot(left_emb, right_emb)
        left_norm = np.linalg.norm(left_emb)
        right_norm = np.linalg.norm(right_emb)
        # Clamp values to prevent floating point error, which results in nan after arccos
        div = np.clip(np.divide(dot, left_norm * right_norm), -1, 1)
        dist_hat = np.arccos(div) * (graph_diameter / (np.pi/2))
    elif measure == "lat-long":
        delta = right_emb - left_emb
        d = 0.5 - np.cos(delta[:, 0])/2 + np.cos(left_emb[:, 0])*np.cos(right_emb[:, 0]) * (1-np.cos(delta[:, 1]))/2
        dist_hat = 2*np.arcsin(np.sqrt(d))
    elif measure == "inv-dot":
        dot = np.dot(left_emb, right_emb)
        left_norm = np.linalg.norm(left_emb)
        right_norm = np.linalg.norm(right_emb)
        div = np.clip(np.divide(dot, left_norm * right_norm), -1, 1)
        dist_hat = -(div-1)*graph_diameter/2
    return dist_hat

# https://github.com/dmlc/dgl/tree/e667545da55017d5dbbd3f243d986506284d3e41/examples/pytorch/node2vec
class ANEDA(nn.Module):
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

    def __init__(self, dataset, num_nodes, embedding_dim, init_embeddings=None, loss_func="mse", measure="norm", norm=2, graph_diameter=1.0, use_sparse=True, device=None):
        super(ANEDA, self).__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.N = num_nodes

        assert loss_func in ["mse", "mre", "mae", "poisson", "custom"]
        if loss_func == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_func == "mae":
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_func == "mre":
            eps = torch.tensor(1e-10, device=device)
            self.loss_fn = lambda pred, true : (torch.abs(pred - true) / torch.maximum(torch.abs(true), eps)).mean()
        elif loss_func == "poisson":
            self.loss_fn = nn.PoissonNLLLoss(log_input=False, eps=1e-07, reduction='mean')
        elif loss_func == "custom":
            def custom_loss(pred, true, delta):
                diff = true - pred
                abs_ = torch.abs(diff)
                delta = torch.quantile(abs_, q=0.99).detach()
                res = torch.where(abs_ <= delta, delta*abs_, (diff**2 + delta**2)/2)
                return res
            self.loss_fn = custom_loss
        
        assert measure in ["norm", "poincare", "hyperboloid", "spherical", "lat-long", "inv-dot"]
        self.measure = measure
        self.norm = norm
        self.graph_diameter = graph_diameter
        
        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)
        print("Initial Embedding Norm:", torch.mean(torch.linalg.norm(init_embeddings, dim=1)))
        if init_embeddings is not None:
            self.embedding.weight = nn.Parameter(init_embeddings)        
        self.embedding = self.embedding.to(device)
        self.device = device
        print(self.embedding.weight.device)
        
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
        #     print("backward") #, torch.sum(torch.isinf(output[0])).item(), torch.sum(torch.isnan(output[0])).item())

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

    def loss(self, samples, return_pred=False, delta=None):
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

        dist_hat = get_distance(left_emb, right_emb, self.measure, self.norm, self.graph_diameter, self.device)

        if delta is not None:
            loss = self.loss_fn(dist_hat, dist, delta)
        else:
            loss = self.loss_fn(dist_hat, dist)

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
            ANEDA training data loader
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


class ANEDAModel(object):
    """
    Wrapper of the ``ANEDA`` class with a ``train`` method.
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

    def __init__(self, dataset, num_nodes, embedding_dim, init_embeddings=None, loss_func="mse", measure="norm", norm=2, graph_diameter=1.0, use_sparse=True, eval_set=None, eval_steps=-1, device='cpu'):
        if device == 'cpu':
            self.device = device
        else:
            # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)

        print(f"...using {self.device}")

        self.model = ANEDA(dataset, num_nodes, embedding_dim, init_embeddings, loss_func, measure, norm, graph_diameter, use_sparse, device)
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.dataset = dataset
        self.eval_set = eval_set
        self.loss_func = loss_func

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

    def _train_step_custom(self, model, loader, optimizer, device, delta):
        model.train()
        total_loss = 0
        losses = []
        for samples in loader:
            samples = samples.to(device)
            optimizer.zero_grad()
            loss_set = model.loss(samples, delta=delta)
            loss = loss_set.mean()
            loss.backward()
            optimizer.step()
            losses.append(loss_set)
            total_loss += loss.mean().item()
        delta = torch.quantile(torch.cat(losses), q=0.99).detach()
        print("Delta:", delta.item())
        return total_loss / len(loader), delta

    @torch.no_grad()
    def _evaluate_step(self, val_loader, evaluate_function=None, config=None, delta=None):
        yhat_list = []
        ytrue_list = []
        val_losses = []
        metric_scores = None
        with torch.no_grad():
            for samples in val_loader:
                samples = samples.to(self.device)
                if self.loss_func == "custom":
                    val_loss, y_val, yhat = self.model.loss(samples, return_pred=True, delta=delta)
                    val_loss = val_loss.mean()
                else:
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

        delta = np.random.rand()
        for i in range(epochs):
            if self.loss_func == "custom":
                loss, delta = self._train_step_custom(self.model, loader, optimizer, self.device, delta)
            else:
                loss = self._train_step(self.model, loader, optimizer, self.device)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    if self.eval_set is not None:
                        validation_loss, val_metrics = self._evaluate_step(val_loader, evaluate_function=evaluate_metrics, config=config, delta=delta)
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


def run_aneda(dataset, num_nodes, init_embeddings=None, eval_set=None, args=None, output_path=None, config=None):
    t_tick = time.time()  ## start measuring running time
    if args is None:
        raise ValueError("need args for ANEDA!")

    # Convert dict to tuple
    ANEDAParams = namedtuple('ANEDAParams', args)
    args = ANEDAParams(**args)

    # train ANEDA
    print("training ANEDA, it will take a while...")
    print("ANEDA arguments:", args)
    trainer = ANEDAModel(dataset,
                            num_nodes=num_nodes,
                            embedding_dim=args.embedding_dim,
                            init_embeddings=init_embeddings,
                            loss_func=args.loss_func,
                            measure=args.measure,
                            norm=args.norm,
                            graph_diameter=config["graph"]["diameter"],
                            eval_set=eval_set,
                            eval_steps=1,
                            device=args.device)

    trainer.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, optimizer=args.optimizer, config=config)

    t_tock = time.time()

    print(f"done training ANEDA, running time = {np.round((t_tock - t_tick) / 60)} minutes.")
    # Calc embedding

    embedding = trainer.embedding().data

    if "cuda" in args.device:
        embedding = embedding.cpu().numpy()

    write_file(output_path, embedding)
    return embedding

def test_aneda(config, embedding, val_set):
    device = config["aneda"]["device"]
    if device != 'cpu':
        device = torch.device(device)
    print(".using device:", device)

    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=True)

    dist_hat_list = []
    dist_true_list = []
    for samples in tqdm(val_loader):
        left, right, dist = samples[:, 0].long(), samples[:, 1].long(), samples[:, 2]
        left_emb = torch.tensor(embedding[left]).to(device)
        right_emb = torch.tensor(embedding[right]).to(device)
        dist_hat = get_distance(left_emb, right_emb, config["aneda"]["measure"], config["aneda"]["norm"], config["graph"]["diameter"], device)

        if device != 'cpu':
            dist = dist.cpu()
            dist_hat = dist_hat.cpu()
        dist = dist.numpy()
        dist_hat = dist_hat.numpy()

        dist_hat_list.append(dist_hat)
        dist_true_list.append(dist)
    
    return evaluate_metrics(np.hstack(dist_true_list).reshape(-1), np.hstack(dist_hat_list).reshape(-1), print_out=False, config=config)