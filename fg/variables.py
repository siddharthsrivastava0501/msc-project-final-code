import torch
from torch import Tensor
from .gaussian import Gaussian
from .graph import Graph

class Variable:
    def __init__(self, id, belief : Gaussian, graph : Graph, num_vars : int = 1, connected_factors = []) -> None:
        self.id = id
        self.belief = belief
        self.num_vars = num_vars

        self.inbox = {}
        self.connected_factors = connected_factors

        self.graph = graph

    @property
    def mean(self) -> Tensor:
        return self.belief.mean

    @property
    def cov(self) -> Tensor:
        return self.belief.cov

    @property
    def eta(self) -> Tensor:
        return self.belief.eta

    @property
    def lmbda(self) -> Tensor:
        return self.belief.lmbda

    # This is so the linter doesn't complain
    def send_initial_messages(self) -> None: pass

    def update_belief(self) -> None:
        '''
        Consume the messages in the inbox to update belief
        '''
        self.belief = Gaussian.zeros_like(self.belief)

        for _, message in self.inbox.items():
            self.belief *= message

        # if not torch.is_nonzero(curr.lmbda): print('We are having a serious problem in the variable')

    def compute_and_send_messages(self) -> None:
        '''
        Equation 2.50, 2.51 in Ortiz (2023)
        '''
        self.update_belief()

        for fid in self.connected_factors:
            if fid == -1: continue

            # Message can be efficiently computed by calculating belief and then
            # dividing with the incoming message?
            msg = self.belief / self.inbox.get(fid, Gaussian.zeros_like(self.belief))

            self.graph.send_msg_to_factor(self.id, fid, msg)

    def __str__(self):
        return f'Variable {self.id} [n = {self.num_vars}, mu={self.mean}, cov={self.cov}]'

class Parameter:
    def __init__(self, id, belief : Gaussian, graph : Graph, connected_factors : list, num_vars : int = 1):
        self.id = id
        self.belief = belief
        self.num_vars = num_vars

        self.inbox = {}
        self.connected_factors = connected_factors
        self.graph = graph

    @property
    def mean(self) -> Tensor:
        return self.belief.mean

    @property
    def cov(self) -> Tensor:
        return self.belief.cov

    @property
    def eta(self) -> Tensor:
        return self.belief.eta

    @property
    def lmbda(self) -> Tensor:
        return self.belief.lmbda

    def update_belief(self) -> None:
        '''
        Consume the messages in the inbox to update belief
        '''
        self.belief = Gaussian.zeros_like(self.belief)

        for _, message in self.inbox.items():
            self.belief *= message

        # if not torch.is_nonzero(curr.lmbda): print('We Hebben Een Serieus Probleem in the parameter')

    def send_initial_messages(self) -> None:
        for fid in self.connected_factors:
            self.graph.send_msg_to_factor(self.id, fid, self.belief.clone())

    def compute_and_send_messages(self) -> None:
        self.update_belief()

        for fid in self.connected_factors:
            if fid == -1: continue

            msg = self.belief / self.inbox.get(fid, Gaussian.zeros_like(self.belief))

            self.graph.send_msg_to_factor(self.id, fid, msg)

    def __str__(self):
        return f'Parameter {self.id} [n = {self.num_vars}, mu={self.mean}, cov={self.cov}]'
