"""
Solving 2d Euler equations
u_t + f(u)_x + g(u)_y = 0
where
u = (rho, rho*u, rho*v, E)
f(u) = (rho*u, rho*u^2+p, rho*u*v, (E+p)*u)
g(u) = (rho*v, rho*u*v, rho*v^2+p, (E+p)*v)
E = 1/2*rho*(u^2+v^2)+rho*e
p = (gamma-1)*rho*e
"""
# %% import modules
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

import flax.linen as nn
from flax.training import train_state
import optax

from typing import List, Tuple
import time
import h5py

# %% prepare data
# Example: vaccum
x_l, x_r = -1, 1
y_d, y_u = -1, 1
T = 0.15
N_x, N_y, N_t = 200, 200, 150

rho_l, rho_r = 1, 1
u_l, u_r = -2, 2
v_l, v_r = 0, 0
p_l, p_r = 0.4, 0.4
gamma = 1.4

N_res = 10000
N_ic = 1000
batch_size = 1000
num_epochs = 20000
learning_rate = 1e-3

num_layer = 4
num_node = 100
layers = [3] + num_layer * [num_node] + [4]

lambda_eqn, lambda_ic = 1, 100


def intialize(x: np.ndarray, y: np.ndarray):
    L_y, L_x = x.shape
    rho_grid, u_grid, v_grid, p_grid = (
        np.zeros_like(x),
        np.zeros_like(x),
        np.zeros_like(x),
        np.zeros_like(x),
    )
    rho_grid[x < 0], rho_grid[x >= 0] = rho_l, rho_r
    u_grid[x < 0], u_grid[x >= 0] = u_l, u_r
    v_grid[x < 0], v_grid[x >= 0] = v_l, v_r
    p_grid[x < 0], p_grid[x >= 0] = p_l, p_r
    return rho_grid, u_grid, v_grid, p_grid


# PDE residual
x = np.random.uniform(x_l, x_r, N_res)
y = np.random.uniform(y_d, y_u, N_res)
t = np.random.uniform(0, T, N_res)
perm = lambda x: np.random.permutation(x)
X_res = np.stack((perm(t), perm(x), perm(y)), 1)
del t, x, y
res_idx = np.random.choice(len(X_res), size=N_res, replace=False)
X_res = X_res[res_idx]

# Initial Condition
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.zeros_like(x)
rho_grid, u_grid, v_grid, p_grid = intialize(x_grid, y_grid)
rho, u, v, p = rho_grid.flatten(), u_grid.flatten(), v_grid.flatten(), p_grid.flatten()
X_ic = np.stack((t, x, y, rho, u, v, p), 1)
del t, x, y, rho, u, v, p
ic_idx = np.random.choice(len(X_ic), size=N_ic, replace=False)
X_ic = X_ic[ic_idx]

# Final points
x, y = np.linspace(x_l, x_r, N_x), np.linspace(y_d, y_u, N_y)
x_grid, y_grid = np.meshgrid(x, y)
x, y = x_grid.flatten(), y_grid.flatten()
t = np.ones_like(x) * T
X_final = np.stack((t, x, y), 1)

# %% define model and model state
class MLP(nn.Module):
    layers: List[int]

    @nn.compact
    def __call__(self, x):
        for nodes in self.layers[1:-1]:
            x = jnp.sin(
                nn.Dense(nodes, kernel_init=jax.nn.initializers.glorot_normal())(x)
            )
        x = nn.Dense(self.layers[-1])(x)
        return x


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    net = MLP(layers)
    params = net.init(rng, jnp.ones([1, layers[0]]))["params"]
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=optimizer
    )


rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
# in this step the model and its state are created
state = create_train_state(rng, learning_rate)
del rng


# %% define loss
def mse(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)


def euler_eqn_fn(net_fn):
    rho, u, v, p = (
        lambda x: net_fn(x)[0],
        lambda x: net_fn(x)[1],
        lambda x: net_fn(x)[2],
        lambda x: net_fn(x)[3],
    )
    rho_e = lambda x: p(x) / (gamma - 1)
    E = lambda x: 0.5 * rho(x) * (u(x) ** 2 + v(x) ** 2) + rho_e(x)
    # components
    # NOTE 1
    u1_t = lambda x: grad(rho)(x)[0]
    f1_x = lambda x: grad(lambda x: rho(x) * u(x))(x)[1]
    g1_y = lambda x: grad(lambda x: rho(x) * v(x))(x)[2]
    # NOTE 2
    u2_t = lambda x: grad(lambda x: rho(x) * u(x))(x)[0]
    f2_x = lambda x: grad(lambda x: rho(x) * u(x) ** 2 + p(x))(x)[1]
    g2_y = lambda x: grad(lambda x: rho(x) * u(x) * v(x))(x)[2]
    # NOTE 3
    u3_t = lambda x: grad(lambda x: rho(x) * v(x))(x)[0]
    f3_x = lambda x: grad(lambda x: rho(x) * u(x) * v(x))(x)[1]
    g3_y = lambda x: grad(lambda x: rho(x) * v(x) ** 2 + p(x))(x)[2]
    # NOTE 4
    u4_t = lambda x: grad(E)(x)[0]
    f4_x = lambda x: grad(lambda x: (E(x) + p(x)) * u(x))(x)[1]
    g4_y = lambda x: grad(lambda x: (E(x) + p(x)) * v(x))(x)[2]
    # equations
    eq1 = lambda x: u1_t(x) + f1_x(x) + g1_y(x)
    eq2 = lambda x: u2_t(x) + f2_x(x) + g2_y(x)
    eq3 = lambda x: u3_t(x) + f3_x(x) + g3_y(x)
    eq4 = lambda x: u4_t(x) + f4_x(x) + g4_y(x)

    return eq1, eq2, eq3, eq4


@jax.jit
def train_step(state, X_res, X_ic):
    """Train for a single step."""

    def loss_fn(params):
        net_fn = lambda x: MLP(layers).apply({"params": params}, x)
        # equation error
        txy = X_res
        eq1, eq2, eq3, eq4 = euler_eqn_fn(net_fn)
        eqn_mass = vmap(eq1, (0))(txy)
        eqn_momentum_x = vmap(eq2, (0))(txy)
        eqn_momentum_y = vmap(eq3, (0))(txy)
        eqn_energy = vmap(eq4, (0))(txy)
        loss_eqn_mass = jnp.mean(eqn_mass ** 2)
        loss_eqn_momentum_x = jnp.mean(eqn_momentum_x ** 2)
        loss_eqn_momentum_y = jnp.mean(eqn_momentum_y ** 2)
        loss_eqn_energy = jnp.mean(eqn_energy ** 2)
        loss_euler = (
            loss_eqn_mass + loss_eqn_momentum_x + loss_eqn_momentum_y + loss_eqn_energy
        )
        # approximation error
        txy, ruvp = X_ic[:, :3], X_ic[:, 3:]
        ruvp_pred = net_fn(txy)
        loss_ic = jnp.mean((ruvp_pred - ruvp) ** 2)
        # total loss
        loss = lambda_eqn * loss_euler + lambda_ic * loss_ic
        return loss, (
            loss,
            loss_eqn_mass,
            loss_eqn_momentum_x,
            loss_eqn_momentum_y,
            loss_eqn_energy,
            loss_ic,
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, losses), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, losses


def mini_batch(N: int, batch_size: int):
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )


def train_epoch(state, data):
    """Train for a single epoch."""
    # unpack data and generate batch
    X_approx, X_res = data
    N_residual = len(X_res)
    N_ic = len(X_ic)
    # batch data for residue constraint
    res_batch_idx_list = mini_batch(N_residual, batch_size)
    # batch data for initial condition
    batch_num = len(res_batch_idx_list)
    batch_size_ic = N_ic // batch_num + 1
    ic_batch_idx_list = mini_batch(N_ic, batch_size_ic)
    # loop through batches
    batch_loss = []
    for ((_, res_idx), (_, ic_idx)) in zip(
        enumerate(res_batch_idx_list),
        enumerate(ic_batch_idx_list),
    ):
        X_res_batch = X_res[res_idx]
        X_ic_batch = X_ic[ic_idx]
        state, losses = train_step(state, X_res_batch, X_ic_batch)
        batch_loss.append(losses)

    # compute mean of metrics across each batch in epoch.
    batch_losses = np.array(jax.device_get(losses))
    epoch_losses = np.mean(batch_losses, axis=0)

    return state, epoch_losses


def save_result(params, X, meta_data, save_path):
    H, W = meta_data["H"], meta_data["W"]
    net_fn = lambda x: MLP(layers).apply({"params": params}, x)
    ruvp = net_fn(X)
    rho, u, v, p = ruvp[:, 0], ruvp[:, 1], ruvp[:, 2], ruvp[:, 3]
    rho, u, v, p = (
        rho.reshape(H, W),
        u.reshape(H, W),
        v.reshape(H, W),
        p.reshape(H, W),
    )
    with h5py.File(save_path, "w") as f:
        f.create_dataset("rho", shape=rho.shape, dtype="float32", data=rho)
        f.create_dataset("u", shape=u.shape, dtype="float32", data=u)
        f.create_dataset("v", shape=v.shape, dtype="float32", data=v)
        f.create_dataset("p", shape=p.shape, dtype="float32", data=p)
        f.create_dataset("X", shape=X.shape, dtype="float32", data=X)


def print_loss(epoch, duration, losses):
    optim_step = epoch * (N_res // batch_size)
    (
        loss,
        loss_eqn_mass,
        loss_eqn_momentum_x,
        loss_eqn_momentum_y,
        loss_eqn_energy,
        loss_ic,
    ) = losses
    loss_Euler = (
        loss_eqn_mass + loss_eqn_momentum_x + loss_eqn_momentum_y + loss_eqn_energy
    )
    print("-" * 50)
    print(
        f"Epoch: {epoch:d}, It: {optim_step:d}, Time: {duration:.2f}s, Learning Rate: {learning_rate:.1e}"
    )
    print(
        f"Epoch: {epoch:d}, It: {optim_step:d}, Loss_sum: {loss:.3e}, Loss_Euler: {loss_Euler:.3e}, Loss_ic: {loss_ic:.3e}"
    )
    print(
        f"Epoch: {epoch:d}, It: {optim_step:d}, Loss_e_m: {loss_eqn_mass:.3e}, Loss_e_x: {loss_eqn_momentum_x:.3e}, Loss_e_y: {loss_eqn_momentum_y:.3e},Loss_e_E: {loss_eqn_energy:.3e}"
    )


N_epochs = 20000
Y_result = []
for epoch in range(1, N_epochs + 1):
    start = time.time()
    state, epoch_losses = train_epoch(state, (X_res, X_ic))
    # if epoch ==2:
    #     profile_result = cProfile.run("model.update(epoch, model.opt_state, X_res, X_ic)")
    #     print(profile_result)
    end = time.time()
    # if epoch % 10 == 0 or epoch == 1:
    print_loss(epoch, end - start, epoch_losses)

save_result(state.params, X_final, {"H": N_y, "W": N_x}, "./test/misc/result.h5")
