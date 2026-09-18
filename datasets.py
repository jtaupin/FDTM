import numpy as np
import numpy.random as rd
from numpy.linalg import norm
from numpy import ndarray


def sphere(n_samples:int, dim:int, X_init:ndarray[float, (...,...)]=None, rng=rd, seed:int=None):
    '''
    Uniformily sampled random points on the `dim`-dimensional sphere embedded in R^(`dim`+1).
    If `rng` is provided, use this generator for reproduceability.
    If `seed` is provided, override `rng` with a new generator for local seeding.
    '''
    if seed is not None : rng = rd.default_rng(seed)
    X = rng.normal(size=(n_samples, dim+1))
    X = X / norm(X, axis=1)[:, np.newaxis]  # Normalise to get uniform on the unit sphere

    if X_init is not None:
        X_init = np.atleast_2d(X_init)
        X[:len(X_init)] = X_init
    return X


def ring(n_samples:int, r:float, X_init:ndarray[float, (...,...)]=None, shortcut:bool=False, only_top:bool=False, rng=rd, seed:int=None):
    '''
    Ring of inner and outer radiuses `1` and `r`.
    Density is higher on the inner radius by a factor `r`.
    If `X_init` is provided, than the beginning of the dataset is replaced with `X_init`.
    If `shortcut` is `True`, then create a line crossing the hole from left to right with `sqrt(n_samples)` points.
    If `only_top` is `True`, then points are only sampled in the upper half.
    If `rng` is provided, use this generator for reproduceability.
    If `seed` is provided, override `rng` with a new generator for local seeding.
    '''
    assert r >= 1
    
    if seed is not None : rng = rd.default_rng(seed)

    # Generate ring
    radius = rng.uniform(1, r, n_samples)
    X = rng.multivariate_normal([0,0], np.identity(2), n_samples)
    X = (X.T / norm(X, axis=1)).T  # Normalise to get uniform on the unit circle
    X = X * np.array([radius, radius]).T  # Multiply by random radius to be on the ring.

    # Replace end of array with a shortcut
    if shortcut:
        n_line = int(n_samples**0.5)
        X_line = np.stack((rng.uniform(-1, 1, n_line), np.zeros(n_line)), axis=-1)  # Shortcut of smaller density
        X[-n_line:] = X_line
    
    # Reverse bottom points to put them at the top instead 
    if only_top:
        X[:, 1] = np.abs(X[:, 1])

    # Replace beginning of array with X_init
    if X_init is not None:
        X_init = np.atleast_2d(X_init)
        X[:len(X_init)] = X_init
    
    return X


def cuboid(n_samples:int, ranges:ndarray[float, (...,)], X_init:ndarray[float, (...,...)]=None, rng=rd, seed:int=None):
    '''
    Uniformily sampled random points on a cuboid with length `2*ranges[k]` in the `k`-th dimension.
    If `rng` is provided, use this generator for reproduceability.
    If `seed` is provided, override `rng` with a new generator for local seeding.
    '''
    if seed is not None : rng = rd.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, len(ranges)))
    for k, r in enumerate(ranges):
        X[:,k] *= r

    if X_init is not None:
        X_init = np.atleast_2d(X_init)
        X[:len(X_init)] = X_init
    return X


def two_balls_line(n_samples:int, proportion:float=0.5, X_init:ndarray[float, (...,...)]=None, rng=rd, seed:int=None):
    '''
    Uniformily sampled random points inside two balls of radius `1` centered at `(-2,0)` and `(2,0)` with a line connecting them.
    `proportion` is the proportion of points in the line.
    If `rng` is provided, use this generator for reproduceability.
    If `seed` is provided, override `rng` with a new generator for local seeding.
    '''
    assert proportion >= 0 and proportion <= 1
    n_line = int(n_samples * proportion) + 1
    
    if seed is not None : rng = rd.default_rng(seed)
    
    X = np.zeros((n_samples, 2))
    X[:n_line, 0] = rng.uniform(-1, 1, size=n_line)  # Line

    # Generate points uniformly in the unit ball by projecting from the sphere in R^4
    X_balls = rng.normal(size=(n_samples-n_line, 4))
    X_balls = X_balls / norm(X_balls, axis=1)[:, np.newaxis]  # Normalise to get uniform on the unit sphere
    X_balls = X_balls[:, :2]  # Project onto first two coordinates
    X_balls[:,0] += rng.integers(0, 2, size=n_samples-n_line) * 4 - 2  # Shift the points to be centered around (-2,0) or (2,0) randomly

    X[n_line:] = X_balls

    if X_init is not None:
        X_init = np.atleast_2d(X_init)
        X[:len(X_init)] = X_init
    
    return X