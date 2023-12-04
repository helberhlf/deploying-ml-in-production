# Import dependencies
import random

import numpy as np
from scipy import optimize as scipy_optimize
from scipy.stats import norm
from sklearn import exceptions
from sklearn import gaussian_process

# Import DL livraries (APIs) bulding up DL pipelines and AutoDL livraries (APIs) for tuning DL pipelines
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.engine import trial as trial_lib

#-------------------------------------------------------

# Customizing the search method of AutoML

# Customizing a Bayesian optimization search method
# Credit: You can find more details about implementing this class in the book Automated Machine Learning in Action
# The full code can be found in the Github Repository of the book: https://github.com/datamllab/automl-in-action-notebooks/blob/master/7.3-Bayesian-Optimization.ipynb
class BayesianOptimizationOracle(oracle_module.Oracle):
    """Bayesian optimization oracle.

    It uses Bayesian optimization with a underlying Gaussian process model.
    The acquisition function used is upper confidence bound (UCB), which can
    be found in the following link:
    https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

    # Arguments
        objective: String or `kerastuner.Objective`. If a string,
          the direction of the optimization (min or max) will be
          inferred.
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has been
            exhausted.
        num_initial_points: (Optional) Int. The number of randomly generated samples
            as initial training data for Bayesian optimization. (If not specified,
            a trick is to use the square root of the dimensionality of the
            hyperparameter space.)
        beta: Float. The balancing factor of exploration and exploitation.
            The larger it is, the more explorative it is.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
    """

    def __init__(
            self,
            objective,
            max_trials,
            beta=2.6,
            acq_type="ucb",
            num_initial_points=None,
            seed=None,
            hyperparameters=None,
            *args,
            **kwargs
    ):
        super(BayesianOptimizationOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            seed=seed,
            *args,
            **kwargs
        )
        # Use 2 as the initial number of random points if not presented.
        self.num_initial_points = num_initial_points or 2
        self.beta = beta
        self.seed = seed or random.randint(1, 1e4)
        self._random_state = np.random.RandomState(self.seed)
        self.gpr = self._make_gpr()
        self.acq_type = acq_type

    def _make_gpr(self):
        return gaussian_process.GaussianProcessRegressor(
            kernel=gaussian_process.kernels.Matern(nu=2.5),
            alpha=1e-4,
            normalize_y=True,
            random_state=self.seed,
        )

    def _vectorize_trials(self):
        x = []
        y = []
        ongoing_trials = set(self.ongoing_trials.values())
        for trial in self.trials.values():
            # Create a vector representation of each Trial's hyperparameters.
            trial_hps = trial.hyperparameters
            vector = []
            for hp in self._nonfixed_space():
                # For hyperparameters not present in the trial (either added
                # after the trial or inactive in the trial), set to default
                # value.
                if (
                    trial_hps.is_active(hp)  # inactive
                    and hp.name in trial_hps.values  # added after the trial
                ):
                    trial_value = trial_hps.values[hp.name]
                else:
                    trial_value = hp.default

                # Embed an HP value into the continuous space [0, 1].
                prob = hp.value_to_prob(trial_value)
                vector.append(prob)

            if trial in ongoing_trials:
                # "Hallucinate" the results of ongoing trials. This ensures that
                # repeat trials are not selected when running distributed.
                x_h = np.array(vector).reshape((1, -1))
                y_h_mean, y_h_std = self.gpr.predict(x_h, return_std=True)
                # Give a pessimistic estimate of the ongoing trial.
                y_h_mean = np.array(y_h_mean).flatten()
                score = y_h_mean[0] + y_h_std[0]
            elif trial.status == "COMPLETED":
                score = trial.score
                # Always frame the optimization as a minimization for
                # scipy.minimize.
                if self.objective.direction == "max":
                    score = -1 * score
            elif trial.status in ["FAILED", "INVALID"]:
                # Skip the failed and invalid trials.
                continue

            x.append(vector)
            y.append(score)

        x = np.array(x)
        y = np.array(y)
        return x, y

    def _vector_to_values(self, vector):
        hps = hp_module.HyperParameters()
        vector_index = 0
        for hp in self.hyperparameters.space:
            hps.merge([hp])
            if isinstance(hp, hp_module.Fixed):
                value = hp.value
            else:
                prob = vector[vector_index]
                vector_index += 1
                value = hp.prob_to_value(prob)

            if hps.is_active(hp):
                hps.values[hp.name] = value
        return hps.values

    def _nonfixed_space(self):
        return [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed)
        ]

    def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

    def _num_completed_trials(self):
        return len([t for t in self.trials.values() if t.status == "COMPLETED"])

    def populate_space(self, trial_id):
        """Fill the hyperparameter space with values.

               Args:
                   trial_id: A string, the ID for this Trial.

               Returns:
                   A dictionary with keys "values" and "status", where "values" is
                   a mapping of parameter names to suggested values, and "status"
                   should be one of "RUNNING" (the trial can start normally), "IDLE"
                   (the oracle is waiting on something and cannot create a trial), or
                   "STOPPED" (the oracle has finished searching and no new trial should
                   be created).
               """
        # Generate enough samples before training Gaussian process.
        completed_trials = [
            t for t in self.trials.values() if t.status == "COMPLETED"
        ]

        # Use 3 times the dimensionality of the space as the default number of
        # random points.
        dimensions = len(self.hyperparameters.space)
        num_initial_points = self.num_initial_points or max(3 * dimensions, 3)
        if len(completed_trials) < num_initial_points:
            return self._random_populate_space()

        # Fit a GPR to the completed trials and return the predicted optimum
        # values.
        x, y = self._vectorize_trials()

        # Ensure no nan, inf in x, y. GPR cannot process nan or inf.
        x = np.nan_to_num(x, posinf=0, neginf=0)
        y = np.nan_to_num(y, posinf=0, neginf=0)

        self.gpr.fit(x, y)

        # Three acquisition functions

        def _upper_confidence_bound(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            return -1 * (mu + self.beta * sigma)

        def _probability_of_improvement(x):
            # calculate the best surrogate score found so far
            x_history, _ = self._vectorize_trials()
            y_pred = self.gpr.predict(x_history, return_std=False)
            y_best = max(y_pred)
            # calculate mean and stdev via surrogate function
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            # calculate the probability of improvement
            z = (mu - y_best) / (sigma + 1e-9)
            prob = norm.cdf(z)
            return -1 * prob

        def _expected_improvement(x):
            # calculate the best surrogate score found so far
            x_history, _ = self._vectorize_trials()
            y_pred = self.gpr.predict(x_history, return_std=False)
            y_best = max(y_pred)
            # calculate mean and stdev via surrogate function
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            # calculate the probability of improvement
            z = (mu - y_best) / (sigma + 1e-9)
            ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
            return -1 * ei

        acq_funcs = {
            "ucb": _upper_confidence_bound,
            "pi": _probability_of_improvement,
            "ei": _expected_improvement,
        }
        # Sampling based on acquisition functions
        optimal_val = float("inf")
        optimal_x = None
        num_restarts = 50
        bounds = self._get_hp_bounds()
        x_seeds = self._random_state.uniform(
            bounds[:, 0], bounds[:, 1], size=(num_restarts, bounds.shape[0])
        )
        for x_try in x_seeds:
            # Sign of score is flipped when maximizing.
            result = scipy_optimize.minimize(
                acq_funcs[self.acq_type], x0=x_try, bounds=bounds, method="L-BFGS-B"
            )
            result_fun = result.fun if np.isscalar(result.fun) else result.fun[0]
            if result_fun < optimal_val:
                optimal_val = result_fun
                optimal_x = result.x

        values = self._vector_to_values(optimal_x)
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _get_hp_bounds(self):
        bounds = [[0, 1] for _ in self._nonfixed_space()]
        return np.array(bounds)

    def get_state(self):
        state = super(BayesianOptimizationOracle, self).get_state()
        state.update(
            {
                "num_initial_points": self.num_initial_points,
                "acq_type": self.acq_type,
                "beta": self.beta,
                "seed": self.seed,
            }
        )
        return state

    def set_state(self, state):
        super(BayesianOptimizationOracle, self).set_state(state)
        self.num_initial_points = state["num_initial_points"]
        self.acq_type = state["acq_type"]
        self.beta = state["beta"]
        self.seed = state["seed"]
        self._random_state = np.random.RandomState(self.seed)
        self.gpr = self._make_gpr()
        self._max_collisions = state["max_collisions"]