import math
import logging
from collections import Counter
from ortools.linear_solver import pywraplp
from sklearn.metrics import ndcg_score
from skopt import gp_minimize
from skopt.space import Real


logger = logging.getLogger(__name__)

class BalancedReranker:
    def __init__(
        self,
        k: int = 50,
        lambda_bounds: tuple = (0.1, 5.0),
        time_limit: int = 60,
        bayes_calls: int = 10,
        initial_points: int = 10,
        slack_weight: float = 0.1,
        imbalance_correction: bool = True,
        verbose: bool = False,
    ):
        # Retained parameters
        self.k = k
        self.base_lower, self.base_upper = lambda_bounds
        self.time_limit = time_limit
        self.bayes_calls = bayes_calls
        self.initial_points = initial_points
        self.slack_weight = slack_weight
        self.imbalance_correction = imbalance_correction

        # Logger setup
        self.logger = logger.getChild(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.logger.info(
            "Initialized: k=%d, λ_bounds=%s, time_limit=%ds, bayes_calls=%d, "
            "init_points=%d, slack_weight=%.3f, imbalance_correction=%s",
            self.k,
            lambda_bounds,
            self.time_limit,
            self.bayes_calls,
            self.initial_points,
            self.slack_weight,
            self.imbalance_correction,
        )


    def set_lambda_bounds(self, lower: float, upper: float):
        self.base_lower, self.base_upper = lower, upper
        self.logger.debug("λ bounds set to [%.3f, %.3f]", lower, upper)

    def set_time_limit(self, seconds: int):
        self.time_limit = seconds
        self.logger.debug("Time limit set to %ds", seconds)

    def set_bayesian_params(self, calls: int, init_points: int):
        self.bayes_calls = calls
        self.initial_points = init_points
        self.logger.debug("Bayes params: calls=%d, init_points=%d", calls, init_points)

    def set_slack_weight(self, weight: float):
        self.slack_weight = weight
        self.logger.debug("Slack weight set to %.3f", weight)

    def set_imbalance_correction(self, enabled: bool):
        self.imbalance_correction = enabled
        self.logger.debug("Imbalance correction set to %s", enabled)


    def _calculate_attention_bias(self, data):
        return [tuple(item) + (1.0 / math.log2(idx + 1),)
                for idx, item in enumerate(data, start=1)]

    def _awrf(self, overall, selected):
        orig_counts = Counter(item[3] for item in overall)
        sel_counts  = Counter(item[3] for item in selected)
        tot_o = sum(orig_counts.values())
        tot_s = sum(sel_counts.values())
        orig_dist = {c: cnt / tot_o for c, cnt in orig_counts.items()}
        sel_dist  = {c: cnt / tot_s for c, cnt in sel_counts.items()}
        return sum(abs(orig_dist[c] - sel_dist.get(c, 0.0))
                   for c in orig_dist) / 2.0

    def _milp(self, data_with_bias, norm_lambda):
        lam = self.base_lower + norm_lambda * (self.base_upper - self.base_lower)
        if self.imbalance_correction:
            counts = Counter(item[3] for item in data_with_bias)
            ratio = max(counts.values()) / min(counts.values()) if counts else 1.0
            lam *= ratio

        solver = pywraplp.Solver.CreateSolver('CBC')
        solver.SetTimeLimit(self.time_limit * 1000)
        n = len(data_with_bias)
        k = min(self.k, n)


        x = {
            (i, j): solver.IntVar(0, 1, f'x[{i},{j}]')
            for i in range(n) for j in range(k)
        }
        discounts = [1.0 / math.log2(j + 2) for j in range(k)]

        obj = solver.Objective()
        for i, item in enumerate(data_with_bias):
            rel, att = item[4], item[6]
            for j in range(k):
                coeff = rel * discounts[j] - lam * att
                obj.SetCoefficient(x[i, j], coeff)
        obj.SetMaximization()


        for i in range(n):
            solver.Add(sum(x[i, j] for j in range(k)) <= 1)

        for j in range(k):
            solver.Add(sum(x[i, j] for i in range(n)) == 1)

        counts = Counter(item[3] for item in data_with_bias)
        total = sum(counts.values())
        fractions = {c: counts[c] / total for c in counts}
        target = {c: int(round(fractions[c] * k)) for c in counts}

        s = sum(target.values())
        for c in counts:
            if s == k: break
            if s > k and target[c] > 0:
                target[c] -= 1; s -= 1
            elif s < k:
                target[c] += 1; s += 1

        for c, inds in {
            c: [i for i,item in enumerate(data_with_bias) if item[3] == c]
            for c in counts
        }.items():
            var = sum(x[i, j] for i in inds for j in range(k))
            slack = solver.NumVar(0, solver.infinity(), f'slack_{c}')
            mn = max(0, target[c] - 1)
            mx = min(counts[c], target[c] + 1)
            solver.Add(var >= mn - slack)
            solver.Add(var <= mx + slack)
            obj.SetCoefficient(slack, -self.slack_weight)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            self.logger.warning('MILP infeasible/time-out → returning top-%d', k)
            return [item for item in data_with_bias[:k]]


        sol = []
        for j in range(k):
            for i in range(n):
                if x[i, j].solution_value() > 0.5:
                    sol.append(data_with_bias[i])
                    break
        return sol

    def optimize_lambda(self, input_data):
        data = self._calculate_attention_bias(input_data)
        ideal = [item[4] for item in data[:self.k]]

        calls = {'n': 0}
        tested = []
        def objective(v):
            calls['n'] += 1
            lam = v[0]
            tested.append(lam)
            ranked = self._milp(data, lam)
            rels = [item[4] for item in ranked]
            ndcg = ndcg_score([ideal], [rels], k=self.k)
            aw = self._awrf(data, ranked)
            score = ndcg * (1 - aw)
            self.logger.info('Bayes %d/%d: λ=%.4f → combined=%.4f',
                              calls['n'], self.bayes_calls, lam, score)
            return -score

        self.logger.info('Starting Bayesian opt: calls=%d, init=%d',
                         self.bayes_calls, self.initial_points)
        result = gp_minimize(
            objective,
            [Real(0.0, 1.0, name='λ')],
            n_calls=self.bayes_calls,
            n_initial_points=self.initial_points,
            acq_func='EI'
        )
        best = result.x[0]
        self.logger.info('Bayes done: λ∈[%.4f,%.4f] → best=%.4f',
                         min(tested), max(tested), best)
        return self._milp(data, best)

    def rerank(self, input_ranking):
        orig = input_ranking[:self.k]
        ideal = [item[4] for item in orig]
        orig_rels = [item[4] for item in orig]
        orig_ndcg = ndcg_score([ideal], [orig_rels], k=self.k)
        orig_aw = self._awrf(input_ranking, orig)
        orig_comb = orig_ndcg * (1 - orig_aw)

        reranked = self.optimize_lambda(input_ranking)
        new_rels = [item[4] for item in reranked]
        new_ndcg = ndcg_score([ideal], [new_rels], k=self.k)
        new_aw = self._awrf(input_ranking, reranked)
        new_comb = new_ndcg * (1 - new_aw)

        if new_comb < orig_comb:
            self.logger.warning(
                'Worse combined (%.4f < %.4f) → returning original.',
                new_comb, orig_comb
            )
            return orig

        self.logger.info('Orig IDs: %s', [i[0] for i in orig])
        self.logger.info('New  IDs: %s', [i[0] for i in reranked])
        return reranked
