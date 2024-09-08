import torch
from torch import Tensor
from .graph import Graph
from .gaussian import Gaussian
from .functions import h_dXdt, h_dYdt, dEdt, dIdt
import numpy as np
import random

class ObservationFactor:
    def __init__(self, factor_id, var_id, z, lmbda_in, graph : Graph, huber = False) -> None:
        self.factor_id = factor_id
        self.var_id = var_id

        self.z = z
        self.Et_z, self.It_z = self.z

        self.lmbda_in = lmbda_in

        J = torch.eye(self.lmbda_in.shape[0])

        # Equation 2.46, 2.47 in Ortiz (2023)
        self.belief = Gaussian.from_canonical((J.T @ lmbda_in) @ z, (J.T @ lmbda_in) @ J)

        # Huber threshold
        self.N_sigma = torch.sqrt(self.lmbda_in[0,0])

        self.inbox = {}

        self.graph = graph

        self.huber = huber

    def update_belief(self) -> None: pass

    def compute_and_send_messages(self) -> None:
        kR = 1.

        message = self.belief * kR
        self.graph.send_msg_to_variable(self.factor_id, self.var_id, message)

    def __str__(self) -> str:
        return f'Obs: [{self.factor_id} -- {self.var_id}], z = {self.z}'


class PriorFactor:
    def __init__(self, factor_id, var_id, z, lmbda_in, graph : Graph, huber = False) -> None:
        self.factor_id = factor_id
        self.var_id = var_id

        self.z = z
        self.lmbda_in = lmbda_in

        self.J = torch.eye(self.lmbda_in.shape[0])

        # Equation 2.46, 2.47 in Ortiz (2023)
        self.belief = Gaussian.from_canonical((self.J.T @ lmbda_in) @ z, (self.J.T @ lmbda_in) @ self.J)

        # Huber threshold
        self.N_sigma = torch.sqrt(self.lmbda_in[0,0])

        self.inbox = {}

        self.graph = graph

    def update_belief(self) -> None: 
        self.belief = Gaussian.from_canonical((self.J.T @ self.lmbda_in) @ self.z, (self.J.T @ self.lmbda_in) @ self.J) 

    def compute_and_send_messages(self) -> None:
        kR = 1.

        message = self.belief * kR
        self.graph.send_msg_to_variable(self.factor_id, self.var_id, message)

    def __str__(self) -> str:
        return f'Prior: [{self.factor_id} -- {self.var_id}], z = {self.z}'


class DynamicsFactor:
    '''
    Represents a dynamics factor that enforces dynamics between `Et_id` (left) and `Etp_id` (right),
    and is also connected to learnable parameters given by `parameters`.
    '''
    def __init__(self, Vt_id, Vtp_id, Sigma_id, lmbda_in : Tensor, factor_id, graph : Graph, huber = False, connected_params = [], mode = 'WC') -> None:
        self.Vt_id, self.Vtp_id = Vt_id, Vtp_id
        self.Sigma_id = Sigma_id
        
        self.lmbda_in = lmbda_in
        self.factor_id = factor_id
        self.graph : Graph = graph

        self.parameters = connected_params

        self.N_sigma = torch.sqrt(lmbda_in)
        self.z = 0

        self.inbox = {}

        # Used for message damping, see Ortiz (2023) 3.4.6
        self._prev_messages = {}

        self._connected_vars = [Vt_id, Vtp_id, Sigma_id] + list(self.parameters)

        self.huber = huber

        self.mode = mode

    def _h_fn(self, Et, It, Etp, Itp, E_ext, I_ext, a, b, c, d, P, Q):
        h_ext = Etp - (Et + 0.05*(dEdt(Et, It, E_ext, a, b, P)))
        h_inh = Itp - (It + 0.05*(dIdt(Et, It, I_ext, c, d, Q)))
        return torch.concat([h_ext, h_inh], dim=1)
    
    def hopf_h_fn(self, Et, It, Etp, Itp, X_ext, Y_ext, a, omega):
        h_ext = Etp - (Et + 0.01*(h_dXdt(Et, It, a, omega, X_ext)))
        h_inh = Itp - (It + 0.01*(h_dYdt(Et, It, a, omega, Y_ext)))
        return torch.concat([h_ext, h_inh], dim=1)
    

    def linearise(self) -> Gaussian:
        '''
        Returns the linearised Gaussian factor based on equations 2.46 and 2.47 in Ortiz (2023)
        '''

        # Extracts the means of all the beliefs of our adj.
        # parameters and gets them ready for autograd
        connected_variables = []
        for i in self._connected_vars:
            mean = self.graph.get_var_belief(i).mean.detach().clone()
            if mean.numel() > 1: #nD beliefs
                for j in range(mean.numel()):
                    connected_variables.append(mean[j].reshape(1, 1).requires_grad_(True))
            else: #1D beliefs
                connected_variables.append(mean.reshape(1, 1).requires_grad_(True))

        Et_mu, It_mu = connected_variables[0:2]
        Etp_mu, Itp_mu = connected_variables[2:4]
        E_ext, I_ext = connected_variables[4:6]


        if self.mode == 'WC':
            a,b,c,d,P,Q = connected_variables[6:]
            self.h = self._h_fn(Et_mu, It_mu, Etp_mu, Itp_mu, E_ext, I_ext, a, b, c, d, P, Q)
            J = torch.concat(torch.autograd.functional.jacobian(self._h_fn, (Et_mu, It_mu, Etp_mu, Itp_mu, E_ext, I_ext, a, b, c, d, P, Q)), 0)[..., 0, 0].T
        else:
            a,omega = connected_variables[6:]
            self.h = self.hopf_h_fn(Et_mu, It_mu, Etp_mu, Itp_mu, E_ext, I_ext, a, omega)
            J = torch.concat(torch.autograd.functional.jacobian(self.hopf_h_fn, (Et_mu, It_mu, Etp_mu, Itp_mu, E_ext, I_ext, a, omega)), 0)[..., 0, 0].T 

        x0 = torch.concat([v for v in connected_variables], dim=0)

        # Have to transpose the h here because otherwise the dimensions don't line up?
        eta = (J.T @ self.lmbda_in) @ (-self.h.T + J @ x0)
        lmbda = (J.T @ self.lmbda_in) @ J 

        return Gaussian.from_canonical(eta.detach(), lmbda.detach())

    def compute_huber(self) -> float:
        # Equation 3.16 in Ortiz (2023)
        r = self.z - self.h
        M = torch.sqrt(r @ self.lmbda_in @ r)

        # Equation 3.20 in Ortiz (2023)
        if M > self.N_sigma and self.huber:
            kR = (2 * self.N_sigma / M) - (self.N_sigma**2 / M**2)
            kR = kR.item()
        else:
            kR = 1.

        return kR

    def _compute_message_to_i(self, i, beta = 1e-1) -> Gaussian:
        '''
        Compute message to variable at index i in `self._vars`,
        All of this is eqn 8 from 'Learning in Deep Factor Graphs with Gaussian Belief Propagation'
        '''
        linearised_factor = self.linearise()

        product = Gaussian.zeros_like(linearised_factor)

        # Build our message product by adding corresponding eta and lambda
        # in product
        k = 0
        for j, id in enumerate(self._connected_vars):
            if j != i:
                in_msg = self.inbox.get(id, Gaussian.from_canonical(torch.tensor([0.]), \
                    torch.tensor([0.])))

                # Element 0 and 1 in self._connected_vars will be the
                # EI oscillator vars, and they each have a 2D Gaussian as their belief
                # since they encode Et, It and Etp, Itp respectively. Therefore,
                # we have to correctly offset our product Gaussian with 2 if
                # our j is at the 0th or 1st element. Otherwise just continue as
                # normal.
                offset = in_msg.eta.numel()
                product.eta[k : k+offset] += in_msg.eta
                product.lmbda[k : k+offset, k : k+offset] += in_msg.lmbda

                k += offset
            else:
                k += self.graph.var_nodes[self._connected_vars[i]].num_vars
                # k += 2 if i in [0,1] else 1
 
        factor_product = linearised_factor * product

        start_idx = 0
        for k in range(i):
            start_idx += self.graph.var_nodes[self._connected_vars[k]].num_vars

        idx_to_marginalise = list(range(start_idx, start_idx + self.graph.var_nodes[self._connected_vars[i]].num_vars))

        marginal = factor_product.marginalise(idx_to_marginalise)

        kR = 1.
        marginal *= kR

        prev_msg = self._prev_messages.get(i, Gaussian.zeros_like(marginal))
        damped_factor = (marginal * beta) * (prev_msg * (1 - beta))

        # Store previous message
        self._prev_messages[i] = damped_factor

        return damped_factor

    def compute_and_send_messages(self, damping = 0.7) -> None:
        for i, var_id in enumerate(self._connected_vars):

            if random.uniform(0,1) < damping: 
                msg = self._compute_message_to_i(i)
                self.graph.send_msg_to_variable(self.factor_id, var_id, msg)

    def __str__(self):
        return f'Dynamics: [{self.Vt_id} -- {self.Vtp_id}], z = {self.z}' 


class AggregationFactor:
    def __init__(self, factor_id, region_id, input_id, C, lmbda_in : Tensor, graph : Graph, connected_regions = []):
        self.lmbda_in = lmbda_in
        self.factor_id = factor_id
        self.graph : Graph = graph

        self.region_id = region_id
        self.input_id = input_id
        self.C = C

        self.connected_regions = connected_regions

        self.z = 0

        self.inbox = {}

        # Used for message damping, see Ortiz (2023) 3.4.6
        self._prev_messages = {}

        self._connected_vars = [self.input_id] + self.connected_regions

    # h() = \Sigmat - \sum (Cnp Xp) where Xp = [E, I] is our oscillator
    def _h_fn(self, Sigma_Et, Sigma_It, *args):
        E_sum, I_sum = 0., 0.

        E = args[::2]
        I = args[1::2]

        for i, region in enumerate(self.connected_regions):
            # Extract the actual region ID of the oscillator by partitioning on 'r'
            # and getting whatever string comes after that
            r_id = int(region.partition('r')[2])
            E_sum += self.C[self.region_id, r_id] * E[i]
            I_sum += self.C[self.region_id, r_id] * I[i]

        return torch.concat([Sigma_Et - E_sum, Sigma_It - I_sum], dim=1)


    def linearise(self) -> Gaussian:

        # Extract the E and I from all our connected regions
        connected_variables = []
        for i in self._connected_vars:
            mean = self.graph.get_var_belief(i).mean.detach().clone()
            if mean.numel() > 1: #nD beliefs
                for j in range(mean.numel()):
                    connected_variables.append(mean[j].reshape(1, 1).requires_grad_(True))
            else: #1D beliefs
                connected_variables.append(mean.reshape(1, 1).requires_grad_(True))
        
        Sigma_Et, Sigma_It = connected_variables[:2]
        other_regions = connected_variables[2:]
        
        self.h = self._h_fn(Sigma_Et, Sigma_It, *other_regions)

        J = torch.concat(torch.autograd.functional.jacobian(self._h_fn, (Sigma_Et, Sigma_It, *other_regions)), 0)[..., 0, 0].T
        x0 = torch.concat([v for v in connected_variables], dim=0)
        
        # Have to transpose the h here because otherwise the dimensions don't line up?
        eta = J.T @ (-self.h.T + J @ x0) * self.lmbda_in
        lmbda = (J.T @ J) * self.lmbda_in

        return Gaussian.from_canonical(eta.detach(), lmbda.detach())
    

    def _compute_message_to_i(self, i, beta = 0.1) -> Gaussian:
        '''
        Compute message to variable at index i in `self._vars`,
        All of this is eqn 8 from 'Learning in Deep Factor Graphs with Gaussian Belief Propagation'
        '''
        linearised_factor = self.linearise()

        product = Gaussian.zeros_like(linearised_factor)

        # Build our message product by adding corresponding eta and lambda
        # in product
        k = 0
        for j, id in enumerate(self._connected_vars):
            if j != i:
                in_msg = self.inbox.get(id, Gaussian.from_canonical(torch.tensor([0.]), \
                    torch.tensor([0.])))
                
                offset = in_msg.eta.numel()
                product.eta[k : k+offset] += in_msg.eta
                product.lmbda[k : k+offset, k : k+offset] += in_msg.lmbda

                k += offset
            else:
                k += self.graph.var_nodes[self._connected_vars[i]].num_vars
 
        factor_product = linearised_factor * product

        start_idx = 0
        for k in range(i):
            start_idx += self.graph.var_nodes[self._connected_vars[k]].num_vars

        idx_to_marginalise = list(range(start_idx, start_idx + self.graph.var_nodes[self._connected_vars[i]].num_vars))

        marginal = factor_product.marginalise(idx_to_marginalise)

        kR = 1.
        marginal *= kR

        prev_msg = self._prev_messages.get(i, Gaussian.zeros_like(marginal))
        damped_factor = (marginal * beta) * (prev_msg * (1 - beta))

        # Store previous message
        self._prev_messages[i] = damped_factor

        return damped_factor
    
    def compute_and_send_messages(self) -> None:
        for i, var_id in enumerate(self._connected_vars):
            msg = self._compute_message_to_i(i)
            self.graph.send_msg_to_variable(self.factor_id, var_id, msg)

    def __str__(self):
        return f'Aggregation {self.factor_id}: [Region: {self.region_id}' 
