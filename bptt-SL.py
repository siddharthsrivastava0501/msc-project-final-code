import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random


# rng controls if we are in 'test' or 'inference' mode so that we get reproducible results when we compute errors.
def dXdt(X, Y, a, omega, rng=None):
    # return (a - X**2 - Y**2)*X - omega*Y + (rng.normal(0, 0.1) if rng else np.random.normal(0, 0.1))
    return (a - X**2 - Y**2)*X - omega*Y + 0.2

def dYdt(X, Y, a, omega, rng=None):
    # return (a - X**2 - Y**2)*Y + omega*X + (rng.normal(0, 0.1) if rng else np.random.normal(0, 0.1))
    return (a - X**2 - Y**2)*Y + omega*X + 0.2

def generate_data(T, dt, a, omega):
    rng = np.random.default_rng(3012)
    X, Y, = [0.0], [0.0]
    Ts = np.arange(0, T + dt, dt)

    for t in range(len(Ts) - 1):
        Xtp = X[t] + dt*(dXdt(X[t], Y[t], a, omega, rng))
        Ytp = Y[t] + dt*(dYdt(X[t], Y[t], a, omega, rng))

        X.append(Xtp)
        Y.append(Ytp)

    return np.stack(X), np.stack(Y)

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SLOscillator(nn.Module):

    def __init__(self, T, dt):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.omega = nn.Parameter(torch.tensor(1.0))

        self.dt = dt
        self.Ts = np.arange(0, T + dt, dt)

    def forward(self):
        X, Y, = [torch.tensor(0.0)], [torch.tensor(0.0)]

        for t in range(len(self.Ts) - 1):
            Xtp = X[t] + self.dt*(dXdt(X[t], Y[t], self.a, self.omega))
            Ytp = Y[t] + self.dt*(dYdt(X[t], Y[t], self.a, self.omega))

            X.append(Xtp)
            Y.append(Ytp)

        return torch.stack(X)

def run_single_experiment(T, dt, EPOCHS, log_every_epochs = 50):
    model = SLOscillator(T, dt)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-2)

    # GT data
    a_GT = np.random.uniform(2, 8)
    omega_GT = np.random.uniform(2, 8) 


    X_gt, _ = generate_data(T, dt, a_GT, omega_GT)
    X_gt = torch.from_numpy(X_gt)

    criterion = nn.MSELoss()

    a_errs = []
    omega_errs = []
    signal_errs = []

    # Start training using BPTT
    for epoch in range(1, EPOCHS+1):

        optim.zero_grad()

        X_hat = model.forward()

        loss = criterion(X_hat.float(), X_gt.float())

        loss.backward()
        optim.step()

        if epoch % log_every_epochs == 0 or epoch == 1:
            a_err = np.abs(a_GT - model.a.item()) / a_GT
            omega_err = np.abs(omega_GT - model.omega.item()) / omega_GT
            signal_err = ((X_gt.numpy() - X_hat.detach().numpy())**2).mean()

            a_errs.append(a_err)
            omega_errs.append(omega_err)
            signal_errs.append(signal_err)

    return a_errs, omega_errs, signal_errs

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
    plt.title(f'Error with 95% Confidence Interval: {parameter} - {num_experiments} experiments')
    plt.legend()

    plt.savefig(f'./figures/BPTT-fixednoise_err_{parameter}.png')


if __name__ == '__main__':
    seed_everything(42)

    T, dt = 4.0, 0.01
    EPOCHS = 500
    log_every_epochs = 20
    NUM_EXPERIMENTS = 50

    a_results = np.zeros(shape=(NUM_EXPERIMENTS, EPOCHS // log_every_epochs + 1))
    omega_results = np.zeros(shape=(NUM_EXPERIMENTS, EPOCHS // log_every_epochs + 1))
    signal_results = np.zeros(shape=(NUM_EXPERIMENTS, EPOCHS // log_every_epochs + 1))

    for i in range(NUM_EXPERIMENTS):
        print(f'Running Experiment {i}')
        a_errs, omega_errs, signal_errs = run_single_experiment(T, dt, EPOCHS, log_every_epochs)

        a_results[i] = a_errs
        omega_results[i] = omega_errs
        signal_results[i] = signal_errs

    plot_95_interval(omega_results, NUM_EXPERIMENTS, 'omega')
    plot_95_interval(a_results, NUM_EXPERIMENTS, 'a')
    plot_95_interval(signal_results, NUM_EXPERIMENTS, 'signal')
    