from fg.gaussian import Gaussian
from fg.graph import Graph
from torch import Tensor
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

# rng controls if we are in 'test' or 'inference' mode so that we get reproducible results when we compute errors.
def dXdt(X, Y, a, omega, P, rng=None):
    # return (a - X**2 - Y**2)*X - omega*Y + (rng.normal(0, 0.1) if rng else np.random.normal(0, 0.1))
    return (a - X**2 - Y**2)*X - omega*Y + P

def dYdt(X, Y, a, omega, Q, rng=None):
    # return (a - X**2 - Y**2)*Y + omega*X + (rng.normal(0, 0.1) if rng else np.random.normal(0, 0.1))
    return (a - X**2 - Y**2)*Y + omega*X + Q

def generate_data(T, dt, a, omega, P, Q):
    rng = np.random.default_rng(3012)
    X, Y, = [0.0], [0.0]
    Ts = np.arange(0, T + dt, dt)

    for t in range(len(Ts) - 1):
        Xtp = X[t] + dt*(dXdt(X[t], Y[t], a, omega, P))
        Ytp = Y[t] + dt*(dYdt(X[t], Y[t], a, omega, Q))

        X.append(Xtp)
        Y.append(Ytp)

    return np.stack(X), np.stack(Y)

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Variable:
    def __init__(self, id, belief : Gaussian, graph : Graph, num_vars : int = 1, connected_factors = []) -> None:
        self.id = id
        self.belief = belief
        self.num_vars = num_vars

        self.inbox = {}
        self.connected_factors = connected_factors

        self.graph = graph

        self.implicit_var = 0.0

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


class ObservationFactor:
    def __init__(self, factor_id, var_id, z, lmbda_in, graph : Graph, huber = False) -> None:
        self.factor_id = factor_id
        self.var_id = var_id

        self.z = z

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
    def __init__(self, Xt_id, Xtp_id, lmbda_in : Tensor, factor_id, graph : Graph, huber = False, connected_params = [], dt = 0.01) -> None:
        self.Xt_id = Xt_id
        self.Xtp_id = Xtp_id
        self.dt = dt

        self.lmbda_in = lmbda_in
        self.factor_id = factor_id
        self.graph : Graph = graph

        self.parameters = connected_params

        self.N_sigma = torch.sqrt(lmbda_in)
        self.z = 0

        self.inbox = {}

        # Used for message damping, see Ortiz (2023) 3.4.6
        self._prev_messages = {}

        self._connected_vars = [Xt_id, Xtp_id] + list(self.parameters)

        self.huber = huber

    def h_fn(self, Xt, Xtp, a, omega, P, Q):        
        Yt = self.graph.var_nodes[self.Xt_id].implicit_var
        Ytp = Yt + self.dt*dYdt(Xt, Yt, a, omega, Q)
        self.graph.var_nodes[self.Xtp_id].implicit_var = Ytp.detach()
        
        h_X = Xtp - (Xt + self.dt*dXdt(Xt, Yt, a, omega, P))
        return h_X

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


        Xt_mu, Xtp_mu = connected_variables[0:2]
        params = connected_variables[2:]

        self.h = self.h_fn(Xt_mu, Xtp_mu, *params)
        J = torch.concat(torch.autograd.functional.jacobian(self.h_fn, (Xt_mu, Xtp_mu, *params)), 0)[..., 0, 0].T 

        x0 =  torch.concat([v for v in connected_variables], dim=0)

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
        return f'Dynamics: [{self.Xt_id} -- {self.Xtp_id} -- {self.Yt_id} -- {self.Ytp_id}], z = {self.z}'
    
def plot_95_interval(results_array, num_experiments, parameter):
    mean_error = np.mean(results_array, axis=0)
    std_error = np.std(results_array, axis=0)
    conf_interval = 1.96 * std_error / np.sqrt(num_experiments)  # 95% CI

    epochs = len(results_array[0])

    plt.figure(figsize=(10, 6))

    plt.plot(mean_error, label='Mean Error', color='blue')

    plt.fill_between(np.arange(epochs), mean_error - conf_interval, mean_error + conf_interval, 
                    color='blue', alpha=0.3, label='95% Confidence Interval')

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'Error with 95% Confidence Interval for {num_experiments} Experiments - {parameter}')
    plt.legend()

    plt.savefig(f'./figures/PIFG_learnablenoise_err_{parameter}.png')

if __name__ == '__main__':
    seed_everything(42)
    
    sigma_obs = 1e-1
    sigma_dynamics = 5e-3
    sigma_prior = 1e1
    T = 4.0
    dt = 0.01
    iters = 70
    nr = 1

    NUM_EXPERIMENTS = 50
    log_every_epochs = 10
    a_results = np.zeros(shape=(NUM_EXPERIMENTS, iters // log_every_epochs + 1))
    omega_results = np.zeros(shape=(NUM_EXPERIMENTS, iters // log_every_epochs + 1))
    p_results = np.zeros(shape=(NUM_EXPERIMENTS, iters // log_every_epochs + 1))
    q_results = np.zeros(shape=(NUM_EXPERIMENTS, iters // log_every_epochs + 1))
    signal_results = np.zeros(shape=(NUM_EXPERIMENTS, iters // log_every_epochs + 1)) 

    # SET UP EXPERIMENTS
    for exp in range(NUM_EXPERIMENTS):
        print(f'Running Experiment {exp}')
        a_GT = np.random.uniform(2, 8)
        omega_GT = np.random.uniform(2, 8) 
        p_GT = np.random.uniform(0, 1)
        q_GT = np.random.uniform(0, 1)
        X_gt, _ = generate_data(T, dt, a_GT, omega_GT, p_GT, q_GT)

        a_errs = []
        omega_errs = []
        p_errs = []
        q_errs = []
        signal_errs = []

        factor_graph = Graph(nr)
        time = torch.arange(0, len(X_gt), 1)

        param_list = ['a', 'omega', 'P', 'Q']

        # ----------- Construct Factor Graph ----------- #
        # Create X variable and obs. factor for X
        for t in range(len(time)):
            factor_graph.var_nodes[f'X_t{t}'] = Variable(
                id       = f'X_t{t}',
                belief   = Gaussian(torch.tensor([[0.1]]), torch.tensor([[0.2]])),
                graph    = factor_graph,
                num_vars = 1,
                connected_factors = [(f'osc_t{t}', f'osc_t{t+1}') if t+1 < len(time) else -1] +  [(f'osc_t{t-1}', f'osc_t{t}') if t > 0 else -1]
            )

            factor_graph.factor_nodes[f'obs_t{t}'] = ObservationFactor(
                factor_id = f'obs_t{t}',
                var_id    = f'X_t{t}',
                z         = torch.tensor([X_gt[t]]).float(),
                lmbda_in  = torch.tensor([[sigma_obs ** -2]]),
                graph     = factor_graph
            )

        # Create our learnable parameters
        for p in param_list:
            p_id = f'p({p})'
            factor_graph.param_ids.append(p_id)
            factor_graph.var_nodes[p_id] = Parameter(
                id     = p_id,
                belief = Gaussian(torch.tensor([[1.0]]), torch.tensor([[sigma_prior ** 2.]])),
                graph  = factor_graph,
                connected_factors = [(f'osc_t{t}', f'osc_t{t+1}') for t in range(len(time)-1)]
            )

            # Add priors to those parameters
            factor_graph.factor_nodes[f'{p_id}_prior'] = PriorFactor(
                factor_id = f'{p_id}_prior',
                var_id = p_id,
                z = torch.tensor([[1.0]]).T,
                lmbda_in = torch.diag(torch.tensor([sigma_prior ** -2])),
                graph = factor_graph
            )

        # Create our dynamics factors
        for t in range(len(time)):
            if t+1 < len(time):
                dyn_id = (f'osc_t{t}', f'osc_t{t+1}')
                factor_graph.factor_nodes[dyn_id] = DynamicsFactor(
                    Xt_id  = f'X_t{t}',
                    Xtp_id = f'X_t{t+1}',
                    lmbda_in = torch.tensor([[sigma_dynamics ** -2]]),
                    factor_id = dyn_id,
                    graph = factor_graph,
                    connected_params = [f'p({p})' for p in param_list],
                    dt = dt
                )


        # ----------- Run GBP ----------- #
        for iter in range(iters+1):
            if iter % log_every_epochs == 0:
                a_learnt = factor_graph.var_nodes['p(a)'].belief.mean.item()
                omega_learnt = factor_graph.var_nodes['p(omega)'].belief.mean.item()
                p_learnt = factor_graph.var_nodes['p(P)'].belief.mean.item()
                q_learnt = factor_graph.var_nodes['p(Q)'].belief.mean.item()

                # Recreate signal
                X_hat, _ = generate_data(T, dt, a_learnt, omega_learnt, p_learnt, q_learnt)

                a_err = np.abs(a_GT - a_learnt) / a_GT
                omega_err = np.abs(omega_GT - omega_learnt) / omega_GT
                p_err = np.abs(p_GT - p_learnt) / p_GT
                q_err = np.abs(q_GT - q_learnt) / q_GT
                signal_err = ((X_gt - X_hat)**2).mean()

                a_errs.append(a_err)
                omega_errs.append(omega_err)
                p_errs.append(p_err)
                q_errs.append(q_err)
                signal_errs.append(signal_err)

                print(a_GT, a_learnt, omega_GT, omega_learnt, p_GT, p_learnt, q_GT, q_learnt)

            if iter == 0:
                factor_graph.update_all_observational_factors()

                for i in factor_graph.var_nodes:
                    curr = factor_graph.var_nodes[i]
                    curr.compute_and_send_messages()

            # Update Factors
            for i in factor_graph.factor_nodes:
                curr = factor_graph.factor_nodes[i]
                curr.compute_and_send_messages() 

            # Posterior as a prior
            factor_graph.update_params()
            for j in factor_graph.param_ids: factor_graph.factor_nodes[f'{j}_prior'].belief = factor_graph.var_nodes[j].belief

            # Update Variables 
            for i in factor_graph.var_nodes:
                if i.__class__.__name__ == 'Variable':
                    curr = factor_graph.var_nodes[i]
                    curr.compute_and_send_messages()

        # Log results
        a_results[exp] = a_errs
        omega_results[exp] = omega_errs
        p_results[exp] = p_errs
        q_results[exp] = q_errs
        signal_results[exp] = signal_errs
    
    plot_95_interval(omega_results, NUM_EXPERIMENTS, 'omega')
    plot_95_interval(a_results, NUM_EXPERIMENTS, 'a')
    plot_95_interval(p_results, NUM_EXPERIMENTS, 'P')
    plot_95_interval(q_results, NUM_EXPERIMENTS, 'Q')
    plot_95_interval(signal_results, NUM_EXPERIMENTS, 'signal')