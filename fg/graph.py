from .gaussian import Gaussian
from typing import Any

class Graph:
    def __init__(self, nr):
        self.var_nodes    = {}
        self.factor_nodes = {}
        self.param_ids    : list[Any] = []
        self.nr           = nr

    def get_var_belief(self, key) -> Gaussian:
        return self.var_nodes[key].belief

    def send_msg_to_factor(self, sender, recipient, msg : Gaussian) -> None:
        self.factor_nodes[recipient].inbox[sender] = msg

    def send_msg_to_variable(self, sender, recipient, msg : Gaussian) -> None:
        self.var_nodes[recipient].inbox[sender] = msg

    def update_dynamics_factor(self, key):
        if isinstance(key, tuple):
            self.factor_nodes[key].compute_and_send_messages()

    def update_params(self):
        for p in self.param_ids:
            self.var_nodes[p].compute_and_send_messages()

    def send_initial_parameter_messages(self):
        for p in self.param_ids:
            self.var_nodes[p].send_initial_messages()

    def update_all_observational_factors(self):
        for key, v in self.factor_nodes.items():
            if v.__class__.__name__ == 'PriorFactor' or v.__class__.__name__ == 'ObservationFactor':
                self.factor_nodes[key].compute_and_send_messages()

    def update_variable_belief(self, key):
        self.var_nodes[key].update_belief()

    def update_all_beliefs(self):
        for key in self.var_nodes:
            self.update_variable_belief(key)

    def prune(self):
        '''
        Remove all the observational and prior factors from the fg so that they
        don't store unnecessary messages and slow down BP / use up memory
        '''
        for k, v in list(self.factor_nodes.items()):
            if v.__class__.__name__ == 'PriorFactor' or v.__class__.__name__ == 'ObservationFactor':
                del self.factor_nodes[k]
