"""
surrogate.py - Machine-learning surrogate for the bend pressure drop.

Purpose
-------
Each transient CFD design point costs ~2.5 h. A surrogate model learns the
mapping (radius, velocity) -> pressure drop from the 9 CFD points so that new
designs can be evaluated in well under a millisecond.

Modelling choices (and why)
---------------------------
* Target = loss coefficient K, not dp.
  dp scales roughly with v^2, so it spans almost an order of magnitude; K is
  nearly constant. Learning the small variation in K is much easier, and we
  recover the pressure drop afterwards with  dp = K * 1/2 * rho * v^2.
  The v^2 physics is therefore built in rather than learned.

* Inputs = (radius, log10 Re).
  Friction/loss correlations vary with log(Re), so log is the natural scale.
  (D_h and fluid properties are fixed, so log Re is equivalent to log v.)

* Normalisation.
  Inputs and target are standardised (zero mean, unit std). The statistics
  are computed on the TRAINING points only inside each cross-validation fold,
  so the held-out point never leaks into the model.

* Small network + regularisation.
  9 data points cannot support a big network. We use one hidden layer of
  8 tanh units (33 trainable weights) with L2 weight decay, and average an
  ensemble of 5 networks started from different random seeds so the answer
  does not hinge on one lucky/unlucky initialisation.

* Leave-one-out cross-validation (LOO-CV).
  With so few points a train/test split would waste data. LOO trains 9 times,
  each time hiding one point and predicting it. Every point is therefore
  predicted by a model that never saw it -> an honest error estimate.
  Note: points on the edge of the design space (radius 0.5 or 3 m,
  velocity 2 or 6 m/s) become EXTRAPOLATIONS when held out, so LOO is a
  pessimistic test.

* Two baselines - if the neural network cannot beat them, it is not worth
  its complexity:
    - linear regression  K = a + b*radius + c*log10(Re)
    - physics baseline   K = constant per radius (mean of its training
      points), i.e. the textbook "K depends on geometry only" assumption.
"""

import time

import numpy as np
import torch
from torch import nn

from . import config
from .sweep import dynamic_pressure
from .validation import reynolds


# --------------------------------------------------------------------------
# Feature engineering and scaling
# --------------------------------------------------------------------------
def make_features(radius, velocity) -> np.ndarray:
    """Build the (n, 2) input matrix [radius, log10(Re)]."""
    # broadcast so a single radius can be paired with many velocities
    radius, velocity = np.broadcast_arrays(
        np.atleast_1d(np.asarray(radius, dtype=float)),
        np.atleast_1d(np.asarray(velocity, dtype=float)))
    return np.column_stack([radius, np.log10(reynolds(velocity))])


class StandardScaler:
    """z = (x - mean) / std, fitted on training data only."""

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)  # avoid /0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, z):
        return z * self.std + self.mean


# --------------------------------------------------------------------------
# Models - both expose fit(X, K) and predict(X) -> K
# --------------------------------------------------------------------------
class LinearBaseline:
    """Ordinary least squares on standardised inputs."""

    def fit(self, x, k):
        self.x_scaler = StandardScaler().fit(x)
        a = np.column_stack([np.ones(len(x)), self.x_scaler.transform(x)])
        self.coef, *_ = np.linalg.lstsq(a, k, rcond=None)
        return self

    def predict(self, x):
        a = np.column_stack([np.ones(len(x)), self.x_scaler.transform(x)])
        return a @ self.coef


class ConstantKPerRadius:
    """Physics baseline: K depends on bend geometry only, not on Re.

    For fully turbulent flow, fitting loss coefficients are roughly
    Reynolds-independent, so the classic engineering estimate is
    dp = K(radius) * 1/2 rho v^2 with one K per bend radius. Here K(radius)
    is the mean K of that radius's training points. No inputs are scaled and
    log Re is ignored on purpose.

    A radius not in the training data (only possible on the design curves,
    never in LOO, where every radius keeps 2 training points) is linearly
    interpolated between the known radii.
    """

    def fit(self, x, k):
        self.radii = np.unique(x[:, 0])
        self.k_mean = np.array([k[x[:, 0] == r].mean() for r in self.radii])
        return self

    def predict(self, x):
        return np.interp(x[:, 0], self.radii, self.k_mean)


class MLPSurrogate:
    """Small ensemble of 2 -> H -> 1 tanh networks trained full-batch with Adam."""

    def __init__(self, hidden=config.HIDDEN_UNITS, n_ensemble=config.N_ENSEMBLE,
                 epochs=config.EPOCHS, lr=config.LEARNING_RATE,
                 weight_decay=config.WEIGHT_DECAY, seed=config.SEED):
        self.hidden, self.n_ensemble = hidden, n_ensemble
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.seed = seed

    def _build(self):
        return nn.Sequential(nn.Linear(2, self.hidden), nn.Tanh(),
                             nn.Linear(self.hidden, 1))

    def fit(self, x, k):
        # Standardise inputs and target using the training data only
        self.x_scaler = StandardScaler().fit(x)
        self.k_scaler = StandardScaler().fit(k.reshape(-1, 1))
        xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        kt = torch.tensor(self.k_scaler.transform(k.reshape(-1, 1)),
                          dtype=torch.float32)

        self.nets = []
        for i in range(self.n_ensemble):
            torch.manual_seed(self.seed + i)       # reproducible, different per member
            net = self._build()
            opt = torch.optim.Adam(net.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            loss_fn = nn.MSELoss()
            for _ in range(self.epochs):           # full batch: all 8-9 points
                opt.zero_grad()
                loss = loss_fn(net(xt), kt)
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, x):
        xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        z = torch.stack([net(xt) for net in self.nets]).mean(dim=0).numpy()
        return self.k_scaler.inverse_transform(z).ravel()


def predict_dp(model, radius, velocity) -> np.ndarray:
    """Pressure drop [Pa] = predicted K * 1/2 rho v^2."""
    velocity = np.atleast_1d(np.asarray(velocity, dtype=float))
    k = model.predict(make_features(radius, velocity))
    return k * dynamic_pressure(velocity)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def leave_one_out(model_factory, x, k):
    """Return the out-of-sample K prediction for every point."""
    preds = np.empty(len(x))
    for i in range(len(x)):
        train = np.arange(len(x)) != i            # all points except i
        model = model_factory().fit(x[train], k[train])
        preds[i] = model.predict(x[i:i + 1])[0]
    return preds


def mape(actual, predicted) -> float:
    """Mean absolute percentage error [%]."""
    return float(np.mean(np.abs((predicted - actual) / actual)) * 100.0)


def time_single_prediction(model, radius=1.5, velocity=4.0, n_repeat=2000):
    """Median wall-clock time [s] of ONE (radius, velocity) -> dp prediction.

    Includes feature building, scaling and the forward pass of every ensemble
    member - i.e. everything a user would wait for.
    """
    predict_dp(model, radius, velocity)            # warm-up (first call is slower)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        predict_dp(model, radius, velocity)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_surrogate(sweep) -> dict:
    torch.set_num_threads(1)   # tiny nets run fastest single-threaded
    x = make_features(sweep["radius"], sweep["velocity"])
    k = sweep["K"].to_numpy()
    q = dynamic_pressure(sweep["velocity"].to_numpy())   # 1/2 rho v^2
    dp = sweep["dp"].to_numpy()

    # Out-of-sample predictions via LOO-CV, converted from K to dp
    dp_loo_mlp = leave_one_out(MLPSurrogate, x, k) * q
    dp_loo_lin = leave_one_out(LinearBaseline, x, k) * q
    dp_loo_ck = leave_one_out(ConstantKPerRadius, x, k) * q

    # Final models trained on all 9 points (used for the design curves)
    mlp = MLPSurrogate().fit(x, k)
    lin = LinearBaseline().fit(x, k)
    ck = ConstantKPerRadius().fit(x, k)

    t_pred = time_single_prediction(mlp)
    t_cfd = config.CFD_RUNTIME_HOURS * 3600.0

    return {
        "mlp": mlp,
        "linear": lin,
        "const_k": ck,
        "dp_loo_mlp": dp_loo_mlp,
        "dp_loo_lin": dp_loo_lin,
        "dp_loo_ck": dp_loo_ck,
        "mape_mlp": mape(dp, dp_loo_mlp),
        "mape_lin": mape(dp, dp_loo_lin),
        "mape_ck": mape(dp, dp_loo_ck),
        "t_pred": t_pred,
        "t_cfd": t_cfd,
        "speedup": t_cfd / t_pred,
    }
