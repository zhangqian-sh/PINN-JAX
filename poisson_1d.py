"""
Solving 1d Poisson equation
-u_xx + u = f(x)
"""
# %% import modules
import numpy as np
import jax.numpy as jnp
from jax import vjp, jit, grad
from jax.example_libraries import optimizers

from functools import partial
import time

# %% initialization
def xavier_init(in_dim, out_dim):
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    W = jnp.array(glorot_stddev * np.random.normal(size=(in_dim, out_dim)))
    b = jnp.zeros(out_dim)
    return W, b


def init_params(layers):
    params = []
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        W, b = xavier_init(in_dim, out_dim)
        params.append({"W": W, "b": b})
    return params


# %% generic pinn function and vector gradient
def pinn_generic(params, X_in):
    X = X_in
    for layer in params[:-1]:
        X = jnp.sin(X @ layer["W"] + layer["b"])
    X = X @ params[-1]["W"] + params[-1]["b"]
    return X


def vgrad(f, x):
    y, vjp_fn = vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


# %% define PINN model
class PINN:
    def __init__(self, layers, weight):
        self.weight_eqn, self.weight_bc = weight
        self.params = init_params(layers)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
        self.opt_state = self.opt_init(self.params)

    # residual
    @partial(jit, static_argnums=(0,))
    def equation(self, params, X):
        """
        -u_xx + u = X
        """
        # define functions
        pinn = partial(pinn_generic, params)
        fu = lambda x: pinn(x)
        dudx = lambda x: vgrad(fu, x)
        d2udx2 = lambda x: vgrad(dudx, x)
        # compute derivatives
        u = fu(X)
        u_xx = d2udx2(X)

        return -u_xx + u - X

    # forward pass
    @partial(jit, static_argnums=(0,))
    def forward(self, params, X):
        # define functions
        pinn = partial(pinn_generic, params)
        fu = lambda x: pinn(x)
        # compute derivatives
        u = fu(X)
        return u

    @partial(jit, static_argnums=(0,))
    def loss_component(self, params, X_res, X_bc):
        x_bc, y_bc = X_bc
        eq1 = self.equation(params, X_res)
        loss_res = jnp.mean(eq1 ** 2)
        # bc
        x_bc_l, x_bc_r = x_bc
        u_l_pred = self.forward(params, x_bc_l)
        u_r_pred = self.forward(params, x_bc_r)
        u_l, u_r = y_bc
        loss_bc = jnp.mean((u_l - u_l_pred) ** 2 + (u_r - u_r_pred) ** 2)
        return loss_res, loss_bc

    def loss(self, params, X_res, X_bc):
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc)
        return self.weight_eqn * loss_res + self.weight_bc * loss_bc

    @partial(jit, static_argnums=(0,))
    def update(self, i, opt_state, X_res, X_bc):
        params = self.get_params(opt_state)
        next_opt_state = self.opt_update(
            i, grad(self.loss)(params, X_res, X_bc), opt_state
        )
        loss_res, loss_bc = self.loss_component(params, X_res, X_bc)
        return next_opt_state, (loss_res, loss_bc)


# %% prepare data
N = 1000
# residual points
X = np.random.uniform(0, 1, size=N).reshape(-1, 1)
X_res = X
# boundary condition
X_bc = (
    (jnp.array(0.0).reshape(-1, 1), jnp.array(1.0).reshape(-1, 1)),
    (jnp.array(0.0).reshape(-1, 1), jnp.array(1.0).reshape(-1, 1)),
)

# %% training
layers = [1, 100, 100, 100, 1]
model = PINN(layers, (1, 1))
[(d["W"].shape, d["b"].shape) for d in model.params]
N_epochs = 1000
Y_result = []
for epoch in range(1, N_epochs + 1):
    start = time.time()
    model.opt_state, losses = model.update(epoch, model.opt_state, X_res, X_bc)
    model.params = model.get_params(model.opt_state)
    end = time.time()
    if epoch % 100 == 0:
        loss_res, loss_bc = losses
        print(
            f"Epoch: {epoch:d}, time: {end-start:.2f}s, Loss_eqn = {loss_res:.3e}, Loss_bc = {loss_bc:.3e}"
        )
        Y = model.forward(model.params, X_res)
        Y_result.append(Y)


# %% Compare with ground truth
def gt(x):
    return x


Y_true = gt(X_res).flatten()
Y_pred = Y_result[-1].flatten()
err = np.mean((Y_pred - Y_true) ** 2) / (np.mean(Y_true) ** 2)
print(f"Relative error = {err:.3e}.")
