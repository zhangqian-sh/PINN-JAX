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
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.example_libraries import optimizers

from functools import partial
import time
import h5py

import itertools

# %% initialization
def glorot_normal(in_dim, out_dim):
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    W = jnp.array(glorot_stddev * np.random.normal(size=(in_dim, out_dim)))
    return W


def init_params(layers):
    params = []
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        W = glorot_normal(in_dim, out_dim)
        b = jnp.zeros(out_dim)
        params.append({"W": W, "b": b})
    return params


# %% functions for training
def pinn_generic(params, X_in):
    X = X_in
    for layer in params[:-1]:
        X = jnp.sin(X @ layer["W"] + layer["b"])
    X = X @ params[-1]["W"] + params[-1]["b"]
    return X


def mini_batch(N: int, batch_size: int):
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )


# %% define PINN model
class PINN:
    def __init__(self, layers, weight, gamma):
        self.weight_eqn, self.weight_ic = weight
        self.params = init_params(layers)
        self.gamma = gamma
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
        self.opt_state = self.opt_init(self.params)
        self.global_step = itertools.count()
        self.log_loss = []

    # forward pass
    def forward(self, params, X):
        ruvp = pinn_generic(params, X)
        return ruvp

    # residual
    def equation(self, params, X):
        """
        input = (t, x, y)
        output = (rho, u, v, p)
        u_t + f(u)_x + g(u)_y = 0
        """
        # define functions
        pinn = lambda x: pinn_generic(params, x)
        rho, u, v, p = (
            lambda x: pinn(x)[0],
            lambda x: pinn(x)[1],
            lambda x: pinn(x)[2],
            lambda x: pinn(x)[3],
        )
        rho_e = lambda x: p(x) / (self.gamma - 1)
        E = lambda x: 0.5 * rho(x) * (u(x) ** 2 + v(x) ** 2) + rho_e(x)
        # components
        # NOTE 1
        u1_t = vmap(lambda x: grad(rho)(x)[0], (0))(X)
        f1_x = vmap(lambda x: grad(lambda x: rho(x) * u(x))(x)[1], (0))(X)
        g1_y = vmap(lambda x: grad(lambda x: rho(x) * v(x))(x)[2], (0))(X)
        # NOTE 2
        u2_t = vmap(lambda x: grad(lambda x: rho(x) * u(x))(x)[0], (0))(X)
        f2_x = vmap(lambda x: grad(lambda x: rho(x) * u(x) ** 2 + p(x))(x)[1], (0))(X)
        g2_y = vmap(lambda x: grad(lambda x: rho(x) * u(x) * v(x))(x)[2], (0))(X)
        # NOTE 3
        u3_t = vmap(lambda x: grad(lambda x: rho(x) * v(x))(x)[0], (0))(X)
        f3_x = vmap(lambda x: grad(lambda x: rho(x) * u(x) * v(x))(x)[1], (0))(X)
        g3_y = vmap(lambda x: grad(lambda x: rho(x) * v(x) ** 2 + p(x))(x)[2], (0))(X)
        # NOTE 4
        u4_t = vmap(lambda x: grad(E)(x)[0], (0))(X)
        f4_x = vmap(lambda x: grad(lambda x: (E(x) + p(x)) * u(x))(x)[1], (0))(X)
        g4_y = vmap(lambda x: grad(lambda x: (E(x) + p(x)) * v(x))(x)[2], (0))(X)

        # equations
        eq1 = u1_t + f1_x + g1_y
        eq2 = u2_t + f2_x + g2_y
        eq3 = u3_t + f3_x + g3_y
        eq4 = u4_t + f4_x + g4_y

        return eq1, eq2, eq3, eq4

    def loss_component(self, params, X_res, X_ic):
        # PDE residual
        eq1, eq2, eq3, eq4 = self.equation(params, X_res)
        loss_eqn_mass, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_energy = (
            jnp.mean(eq1 ** 2),
            jnp.mean(eq2 ** 2),
            jnp.mean(eq3 ** 2),
            jnp.mean(eq4 ** 2),
        )
        loss_eqn = (
            loss_eqn_mass + loss_eqn_momentum_x + loss_eqn_momentum_y + loss_eqn_energy
        )
        # initial condition
        x_ic, y_ic = X_ic[:, :3], X_ic[:, 3:]
        ruvp_pred = self.forward(params, x_ic)
        ruvp = y_ic
        loss_ic = jnp.mean((ruvp_pred - ruvp) ** 2)
        return loss_eqn, loss_ic

    def loss(self, params, X_res, X_ic):
        loss_eqn, loss_ic = self.loss_component(params, X_res, X_ic)
        return self.weight_eqn * loss_eqn + self.weight_ic * loss_ic

    @partial(jit, static_argnums=(0,))
    def loss_evaluation(self, params, X_res, X_ic):
        # PDE residual
        eq1, eq2, eq3, eq4 = self.equation(params, X_res)
        loss_eqn_mass, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_energy = (
            jnp.mean(eq1 ** 2),
            jnp.mean(eq2 ** 2),
            jnp.mean(eq3 ** 2),
            jnp.mean(eq4 ** 2),
        )
        loss_eqn = (
            loss_eqn_mass + loss_eqn_momentum_x + loss_eqn_momentum_y + loss_eqn_energy
        )
        # initial condition
        x_ic, y_ic = X_ic[:, :3], X_ic[:, 3:]
        ruvp_pred = self.forward(params, x_ic)
        ruvp = y_ic
        loss_ic = jnp.mean((ruvp_pred - ruvp) ** 2)
        return loss_eqn, loss_ic

    @partial(jit, static_argnums=(0,))
    def update(self, i, opt_state, X_res, X_ic):
        params = self.get_params(opt_state)
        next_opt_state = self.opt_update(
            i, grad(self.loss)(params, X_res, X_ic), opt_state
        )
        return next_opt_state

    def train(self, X_res: np.ndarray, X_ic: np.ndarray):
        """
        loss_avg shape
        [loss, loss_supervised, loss_eqn_momentum_x, loss_eqn_momentum_y, loss_eqn_mass, loss_bc]
        """
        N_residual = len(X_res)
        N_ic = len(X_ic)
        # batch data for residue constraint
        res_batch_idx_list = mini_batch(N_residual, batch_size)
        # batch data for initial condition
        batch_num = len(res_batch_idx_list)
        batch_size_ic = N_ic // batch_num + 1
        ic_batch_idx_list = mini_batch(N_ic, batch_size_ic)
        # loop through batches
        loss_array = []
        for ((_, res_idx), (_, ic_idx)) in zip(
            enumerate(res_batch_idx_list),
            enumerate(ic_batch_idx_list),
        ):
            # gather and convert data
            X_res_batch = X_res[res_idx]
            X_ic_batch = X_ic[ic_idx]
            # train for a batch
            self.params = self.get_params(self.opt_state)
            self.opt_state = self.update(
                next(self.global_step), self.opt_state, X_res_batch, X_ic_batch
            )
            loss_list = self.loss_evaluation(self.params, X_res_batch, X_ic_batch)
            loss_array.append(loss_list)
        loss_avg = np.mean(np.array(loss_array), axis=0)
        self.log_loss.append(loss_avg)
        return loss_avg

    def save_result(self, X, meta_data, save_path):
        H, W = meta_data["H"], meta_data["W"]
        ruvp = self.forward(self.params, X)
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

N_res = 1000
N_ic = 1000
batch_size = 1000
num_epochs = 20000

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

# train model
model = PINN(layers, (lambda_eqn, lambda_ic), gamma)


N_epochs = 20000
Y_result = []
for epoch in range(1, N_epochs + 1):
    start = time.time()
    losses = model.train(X_res, X_ic)
    # if epoch ==2:
    #     profile_result = cProfile.run("model.update(epoch, model.opt_state, X_res, X_ic)")
    #     print(profile_result)
    end = time.time()
    # if epoch % 10 == 0 or epoch == 1:
    loss_eqn, loss_ic = losses
    print(
        f"Epoch: {epoch:d}, time: {end-start:.3f}s, Loss_eqn = {loss_eqn:.3e}, Loss_ic = {loss_ic:.3e}"
    )

model.save_result(X_final, {"H": N_y, "W": N_x}, "./test/misc/result.h5")
