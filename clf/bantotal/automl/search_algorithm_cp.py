# Import dependencies
import random

import numpy as np
from scipy import optimize as scipy_optimize
from scipy.stats import norm
from sklearn import exceptions
from sklearn import gaussian_process

# Import DL livraries (APIs) bulding up DL pipelines and AutoDL livraries (APIs) for tuning DL pipelines
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import multi_execution_tuner
from keras_tuner.engine import oracle as oracle_module
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
        x, y = [], []
        for trial in self.trials.values(): # Loops através de todos os ensaios
            # Create a vector representation of each Trial's hyperparameters.
            trial_hps = trial.hyperparameters
            vector = []
            nonfixed_hp_space = [
                hp
                for hp in self.hyperparameters.space
                # Grava o hiperparâmetros que não são fixos
                if not isinstance(hp, hp_module.Fixed)
            ]
            for hp in nonfixed_hp_space:
                # For hyperparameters not present in the trial (either added after
                # the trial or inactive in the trial), set to default value.
                if trial_hps.is_active(hp):
                    trial_value = trial_hps.values[hp.name]
                else:
                    # Usa o valor padrão para não utilizado hiperparâmetro
                    trial_value = hp.default

                # Embed an HP value into the continuous space [0, 1].
                prob = hp_module.value_to_cumulative_prob(trial_value, hp)
                vector.append(prob)

            if trial.status == "COMPLETED":
                score = trial.score
                if self.objective.direction == "min":
                    # Unifica o pontuação de avaliação de modo que maior os valores são sempre Melhor
                    score = -1 * score
            else:
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
                value = hp_module.cumulative_prob_to_value(prob, hp)

            if hps.is_active(hp):
                hps.values[hp.name] = value
        return hps.values

    def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}
    def _num_completed_trials(self):
        return len([t for t in self.trials.values() if t.status == "COMPLETED"])
    def populate_space(self, trial_id):
        if self._num_completed_trials() < self.num_initial_points:
            return self._random_populate_space()

        # Update Gaussian process regressor
        x, y = self._vectorize_trials()
        try:
            self.gpr.fit(x, y)
        except exceptions.ConvergenceWarning as e:
            raise e
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
        nonfixed_hp_space = [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed)
        ]
        bounds = []
        for hp in nonfixed_hp_space:
            bounds.append([0, 1])
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
#-------------------------------------------------------

# Use customized evolutionary search algorithm to tune model
# Credit: You can find more details about implementing this class in the book Automated Machine Learning in Action
# The full code can be found in the Github Repository of the book: https://github.com/datamllab/automl-in-action-notebooks/blob/master/7.4-Evolutionary-Search.ipynb
class EvolutionaryOracle(oracle_module.Oracle):
    """Evolutionary search oracle.

        It uses aging evluation algorithm following: https://arxiv.org/pdf/1802.01548.pdf.
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
                as initial training data for Evolutionary search. If not specified,
                a value of 3 times the dimensionality of the hyperparameter space is
                used.
            population_size: (Optional) Int. The number of trials to form the populations.
    candidate_size: (Optional) Int. The number of candidate trials in the tournament
    selection.
            seed: Int. Random seed.
            hyperparameters: HyperParameters class instance.
                Can be used to override (or register in advance)
                hyperparamters in the search space.
    """

    def __init__(
        self,
        objective,
        max_trials,
        num_initial_points=None,
        population_size=None,
        candidate_size=None,
        seed=None,
        hyperparameters=None,
        *args,
        **kwargs
    ):
        super(EvolutionaryOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            seed=seed,
            *args,
            **kwargs
        )
        self.population_size = population_size or 20
        self.candidate_size = candidate_size or 5
        self.num_initial_points = num_initial_points or self.population_size
        self.num_initial_points = max(self.num_initial_points, population_size)
        self.population_trial_ids = []
        self.seed = seed or random.randint(1, 1e4)
        self._seed_state = self.seed
        self._random_state = np.random.RandomState(self.seed)
        self._max_collisions = 100

    def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _num_completed_trials(self):
        return len([t for t in self.trials.values() if t.status == "COMPLETED"])

    def populate_space(self, trial_id):

        if self._num_completed_trials() < self.num_initial_points:
            return self._random_populate_space()

        self.population_trial_ids = self.end_order[-self.population_size :]

        # candidate trial selection
        candidate_indices = self._random_state.choice(
            self.population_size, self.candidate_size, replace=False
        )
        self.candidate_indices = candidate_indices
        candidate_trial_ids = list(
            map(self.population_trial_ids.__getitem__, candidate_indices)
        )

        # get the best candidate based on the performance
        candidate_scores = [
            self.trials[trial_id].score for trial_id in candidate_trial_ids
        ]
        best_candidate_trial_id = candidate_trial_ids[np.argmin(candidate_scores)]
        best_candidate_trial = self.trials[best_candidate_trial_id]

        # mutate the hps of the candidate
        values = self._mutate(best_candidate_trial)

        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}

        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _mutate(self, best_trial):

        best_hps = best_trial.hyperparameters

        # get non-fixed and active hyperparameters in the trial to be mutated
        nonfixed_active_hps = [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed) and best_hps.is_active(hp)
        ]

        # random select a hyperparameter to mutate
        hp_to_mutate = self._random_state.choice(nonfixed_active_hps, 1)[0]

        collisions = 0
        while True:
            hps = hp_module.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and hp.name != hp_to_mutate.name:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values

            # Make sure the new hyperparameters has not been evaluated before
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values

    def get_state(self):
        state = super(EvolutionaryOracle, self).get_state()
        state.update(
            {
                "num_initial_points": self.num_initial_points,
                "population_size": self.population_size,
                "candidate_size": self.candidate_size,
                "seed": self.seed,
                "_max_collisions": self._max_collisions,
            }
        )
        return state

    def set_state(self, state):
        super(EvolutionaryOracle, self).set_state(state)
        self.num_initial_points = state["num_initial_points"]
        self.population_size = state["population_size"]
        self.candidate_size = state["candidate_size"]
        self.population_trial_ids = self.end_order[-self.population_size :]
        self.seed = state["seed"]
        self._random_state = np.random.RandomState(self.seed)
        self._seed_state = self.seed
        self._max_collisions = state["max_collisions"]