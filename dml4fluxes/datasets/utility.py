from torch.utils import data
from jax import vmap, random, jit
from jax import numpy as jnp
from functools import partial


class BootstrapLoader(data.Dataset):
    def __init__(
        self,
        X,
        T,
        y,
        batch_size=128,
        ensemble_size=32,
        fraction=0.8,
        n_samples=None,
        rng_key=random.PRNGKey(1234),
        replace=True,
    ):
        "Initialization"
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.replace = replace
        if n_samples:
            self.bootstrap_size = n_samples
        else:
            self.bootstrap_size = int(self.N * fraction)
        self.key = rng_key
        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        self.X, self.T, self.y, self.indices = vmap(
            self.__bootstrap, (None, None, None, 0)
        )(X, T, y, keys)
        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X, self.y
        )

    @partial(jit, static_argnums=(0,))
    def normalization_constants(self, X, y):
        mu_X, sigma_X = X.mean(0), X.std(0)
        mu_y, sigma_y = jnp.zeros(
            y.shape[1],
        ), jnp.abs(
            y
        ).max(0) * jnp.ones(
            y.shape[1],
        )

        return (mu_X, sigma_X), (mu_y, sigma_y)

    @partial(jit, static_argnums=(0,))
    def __bootstrap(self, X, T, y, key):
        idx = random.choice(key, self.N, (self.bootstrap_size,), replace=self.replace)
        inputs = X[idx, :]
        T = T[idx, :]
        targets = y[idx, :]
        return inputs, T, targets, idx

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, T, y, norm_const):
        "Generates data containing batch_size samples"
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        T = T[idx, :]
        X = X[idx, :]
        y = y[idx, :]
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y
        return X, T, y

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, T, targets = vmap(self.__data_generation, (0, 0, 0, 0, 0))(
            keys, self.X, self.T, self.y, self.norm_const
        )
        return inputs, T, targets


class CustomBootstrapLoader(data.Dataset):
    def __init__(
        self,
        X,
        y,
        batch_size=128,
        ensemble_size=32,
        split=0.8,
        rng_key=random.PRNGKey(1234),
    ):
        #'Initialization'
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.split = split
        self.key = rng_key

        if self.N < self.batch_size:
            self.batch_size = self.N

        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        if split < 1:
            self.data_train, self.data_val = vmap(self.__bootstrap, (None, None, 0))(
                X, y, keys
            )
            (self.X_train, self.y_train) = self.data_train
        else:
            self.data_train, self.data_val = vmap(
                self.__bootstrap_train_only, (None, None, 0)
            )(X, y, keys)
            (self.X_train, self.y_train) = self.data_train

        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X_train, self.y_train
        )

        # For analysis reasons
        self.norm_const_val = vmap(self.normalization_constants, in_axes=(0, 0))(
            *self.data_val
        )

    def normalization_constants(self, X, y):
        mu_X, sigma_X = X.mean(0), X.std(0)
        mu_y, sigma_y = jnp.zeros(
            y.shape[1],
        ), jnp.abs(
            y
        ).max(0) * jnp.ones(
            y.shape[1],
        )

        return (mu_X, sigma_X), (mu_y, sigma_y)

    def __bootstrap(self, X, y, key):
        # TODO Proper Bootstrap is happening outside. In here we take the whole dataset and split it
        idx = random.choice(key, self.N, (self.N,), replace=False)
        idx_train = idx[: jnp.floor(self.N * self.split).astype(int)]
        idx_test = idx[jnp.floor(self.N * self.split).astype(int) :]

        inputs_train = X[idx_train, :]
        targets_train = y[idx_train, :]

        inputs_test = X[idx_test, :]
        targets_test = y[idx_test, :]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    def __bootstrap_train_only(self, X, y, key):
        idx = random.choice(key, self.N, (self.N,), replace=False).sort()

        inputs_train = X[idx]
        targets_train = y[idx]

        inputs_test = X[idx]
        targets_test = y[idx]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, y, norm_const):
        "Generates data containing batch_size samples"
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        X = X[idx, :]
        y = y[idx, :]
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y
        return X, y

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, targets = vmap(self.__data_generation, (0, 0, 0, 0))(
            keys, self.X_train, self.y_train, self.norm_const
        )
        return inputs, targets