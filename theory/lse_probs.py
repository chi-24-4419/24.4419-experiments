from imlmlib.exponential_forgetting import GaussianEFPopulation
from imlmlib.mem_utils import BlockBasedSchedule, experiment
import numpy


Npart = 1
Nitems = 10

schedule = BlockBasedSchedule(Nitems, 5, [2000, 2000, 86400, 2000], repet_trials=3, sigma_t = 1)


def generate_paper_obs_data(schedule):
    population_kwargs = {
        "population_size": Npart,
        "n_items": Nitems,
        "seed": None,
        "mu_a": 1e-2,
        "sigma_a": 1e-6,
        "mu_b": 0.3,
        "sigma_b": 1e-6,
    }

    population_model = GaussianEFPopulation(**population_kwargs)
    data = experiment(population_model, schedule, replications=1)
    nblock = len(schedule.interblock_time) + 1

    data = (
        data[0, 0, ...]
        .transpose(1, 0)
        .reshape(
            population_model.pop_size,
            nblock,
            schedule.nitems * schedule.repet_trials,
        )
    )
    return data.mean(axis=(0, 2)).squeeze()


population_kwargs = {
    "population_size": Npart,
    "n_items": Nitems,
    "seed": 1234,
    "mu_a": 1e-2,
    "sigma_a": 1e-6,
    "mu_b": 0.2,
    "sigma_b": 1e-6,
}

schedule = BlockBasedSchedule(Nitems, 5, [2000, 2000, 86400, 2000], repet_trials=3)

paper_obs = generate_paper_obs_data(schedule)


def to_optimize(population_kwargs, schedule, paper_obs, theta):

    log10_alpha, beta = theta
    alpha = 10**log10_alpha
    # alpha = log10_alpha
    population_kwargs.update({"mu_a": alpha, "mu_b": beta})
    population_model = GaussianEFPopulation(**population_kwargs)

    nblock = len(schedule.interblock_time) + 1
    data = experiment(population_model, schedule, replications=1)
    data = (
        data[0, 0, ...]
        .transpose(1, 0)
        .reshape(
            population_model.pop_size,
            nblock,
            schedule.nitems * schedule.repet_trials,
        )
    )
    predicted_obs = data.mean(axis=(0, 2)).squeeze()

    _diff = predicted_obs - paper_obs
    _sum = predicted_obs + paper_obs

    return numpy.sum(numpy.divide(_diff**2, _sum, out = numpy.zeros(_diff.shape, dtype = numpy.float64), where=_sum!=0))
       

    # return numpy.abs(predicted_obs - paper_obs)

import functools
import scipy.optimize as opti

_opti = functools.partial(to_optimize, population_kwargs, schedule, paper_obs)

# res = opti.least_squares(_opti, [-2, 0.5],  jac = '3-point')
res = opti.differential_evolution(_opti, bounds = [(-6,-.5),(0.01,.99)])
# res = opti.differential_evolution(_opti, bounds = [(10**(-6),10**(-.5)),(0.01,.99)])

import statsmodels

hess = statsmodels.tools.numdiff.approx_hess2(res.x, _opti, epsilon = 1e-1)
hess_inv = numpy.linalg.inv(hess)
rho = numpy.sqrt(hess_inv[1,0]**2/(hess_inv[0,0]*hess_inv[1,1]))
