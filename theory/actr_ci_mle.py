from imlmlib.actr import (
    ACTR,
    identify_actr_from_recall_sequence,
    actr_observed_information_matrix,
)
from imlmlib.mle_utils import CI_asymptotical
import numpy
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

SEED = 789
N_logspace = 3
REPETITIONS = 500
DATA_GEN = True

rng = numpy.random.default_rng(seed=SEED)


def simulate_arbitrary_traj(actr, N):
    recall = []
    times = []
    query_times = []
    for trial in range(N):
        actr.reset()
        nrepet = rng.integers(low=1, high=10, size=(1,))
        _times = numpy.sort(rng.random(size=(int(nrepet),)) * 100)
        times.append(_times)
        actr.update(0, _times)
        query_time = rng.random(size=(1,)) * 100 + _times[-1]
        query_times.append(query_time)
        recall.append(actr.query_item(0, query_time))
    return recall, times, query_times


D_TRUE_VALUE = 0.4
S_TRUE_VALUE = 0.1
TAU_TRUE_VALUE = -0.5
actr = ACTR(1, d=D_TRUE_VALUE, s=S_TRUE_VALUE, tau=TAU_TRUE_VALUE, seed=SEED)
# Data gen
def gen_data(actr_model, N):
    recalls, times, query_times = simulate_arbitrary_traj(actr_model, N)
    recalls = [int(r[0]) for r in recalls]
    deltatis = [qtime - time for time, qtime in zip(times, query_times)]
    return recalls, deltatis


# optimizer --- same options for any N
# optim_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1), (-5, 5), (-5, 5)]}
optim_kwargs = {"method": "BFGS"}

verbose = False
# d, s, tau
guess = (0.5, 0.25, -0.7)


if DATA_GEN:
    results = {}
    for _N in numpy.logspace(1, 3, N_logspace):
        N = int(numpy.round(_N))
        result = numpy.zeros((3, 3, REPETITIONS))
        for i in tqdm(range(REPETITIONS)):
            recalls, deltatis = gen_data(actr, N)
            inference_results = identify_actr_from_recall_sequence(
                recalls,
                deltatis,
                optim_kwargs=optim_kwargs,
                verbose=verbose,
                guess=guess,
                basin_hopping=True,
                basin_hopping_kwargs={"niter": 3},
            )

            if (
                inference_results.lowest_optimization_result.x[0] <= 0
                or inference_results.lowest_optimization_result.x[0] >= 1
                or inference_results.lowest_optimization_result.x[1] <= -5
                or inference_results.lowest_optimization_result.x[1] >= 5
                or inference_results.lowest_optimization_result.x[2] <= -5
                or inference_results.lowest_optimization_result.x[2] >= 5
            ):
                result[..., i] = numpy.nan
                continue

            J = actr_observed_information_matrix(
                recalls, deltatis, *inference_results.lowest_optimization_result.x
            )
            try:
                covar = numpy.linalg.inv(J)
            except numpy.linalg.LinAlgError:
                result[..., i] = numpy.nan
                result[:, 0, i] = inference_results.lowest_optimization_result.x
                continue
            hess_inv = inference_results.lowest_optimization_result.hess_inv
            # test coverage and plot
            cis = CI_asymptotical(
                covar,
                inference_results.lowest_optimization_result.x,
                critical_value=1.96,
            )
            cis_hess_inv = CI_asymptotical(
                hess_inv,
                inference_results.lowest_optimization_result.x,
                critical_value=1.96,
            )

            result[:, 0, i] = inference_results.lowest_optimization_result.x
            TRUE_VALUES = [D_TRUE_VALUE, S_TRUE_VALUE, TAU_TRUE_VALUE]
            for n in range(3):
                if cis[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                    result[n, 1, i] = True
                else:
                    result[n, 1, i] = False
                if cis_hess_inv[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                    result[n, 2, i] = True
                else:
                    result[n, 2, i] = False

        results[str(N)] = result.tolist()
    with open("save_data/actr_asymptotic.json", "w") as _file:
        json.dump(results, _file)
        results = {key: numpy.array(value) for key, value in results.items()}

else:
    with open("save_data/actr_asymptotic.json", "r") as _file:
        results = json.load(_file)
        results = {key: numpy.array(value) for key, value in results.items()}

# print(numpy.nanmean(a=results["100"], axis=2))
# exit()
# xaxis = [float(k) for k in results.keys()]
# coverage_array = numpy.zeros((2, 2, len(xaxis)))
# for n, (key, value) in enumerate(results.items()):
#     coverage_array[0, :, n] = numpy.mean(value[0, :, :], axis=1)
#     coverage_array[1, :, n] = numpy.mean(value[1, :, :], axis=1)

# import seaborn as sb
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(nrows=1, ncols=2)
# fig.suptitle("Coverage of computed confidence intervals")

# sb.lineplot(
#     x=xaxis, y=coverage_array[0, 0, :], ax=axs[0], label=r"$\alpha~\mathrm{coverage}$"
# )
# ax = sb.lineplot(
#     x=xaxis, y=coverage_array[0, 1, :], ax=axs[0], label=r"$\beta~\mathrm{coverage}$"
# )
# ax.set(xscale="log")
# ax.set_xlabel("Sample size")
# ax.set_ylabel("Coverage")
# ax.set_title(r"$\alpha,~\beta$")

# sb.lineplot(
#     x=xaxis,
#     y=coverage_array[1, 0, :],
#     ax=axs[1],
#     label=r"$\log_{10}(\alpha)~\mathrm{coverage}$",
# )
# ax = sb.lineplot(
#     x=xaxis, y=coverage_array[1, 1, :], ax=axs[1], label=r"$\beta~\mathrm{coverage}$"
# )

# ax.set(xscale="log")
# ax.set_xlabel("Sample size")
# ax.set_ylabel("Coverage")
# ax.set_title(r"$\log_{10}(\alpha),~\beta$")
# plt.show()
# plt.savefig("images/CIcoverage_log.pdf")
