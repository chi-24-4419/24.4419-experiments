from imlmlib.ef import (
    ExponentialForgetting,
    identify_ef_from_recall_sequence,
    covar_delta_method_log_alpha,
    ef_observed_information_matrix,
)
from imlmlib.mle_utils import (
    CI_asymptotical,
)
import numpy
from tqdm import tqdm
import pandas
import json
import matplotlib.pyplot as plt


plt.style.use(style="fivethirtyeight")

SEED = 789
N_logspace = 4
REPETITIONS = 1000


def simulate_arbitrary_traj(ef, k_vector, deltas):
    recall = []
    for k, d in zip(k_vector, deltas):
        ef.update(0, 0, N=k)
        recall.append(ef.query_item(0, d))
    return recall


ALPHA_TRUE = 1e-2
BETA_TRUE = 0.4
ef = ExponentialForgetting(1, ALPHA_TRUE, BETA_TRUE, seed=SEED)
rng = numpy.random.default_rng(seed=SEED)
################
GEN_DATA = False

if GEN_DATA:
    results = {}
    for _N in numpy.logspace(1, 4, N_logspace):
        N = int(numpy.round(_N))
        print(N)
        result = numpy.zeros((2, 4, REPETITIONS))
        for i in tqdm(range(REPETITIONS)):
            k_vector = rng.integers(low=0, high=10, size=N)
            deltas = rng.integers(low=0, high=5000, size=N)
            recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
            recall = [rp[0] for rp in recall_probs]
            optim_kwargs = {"method": "BFGS"}

            verbose = False
            guess = (1e-3, 0.5)
            inference_results = identify_ef_from_recall_sequence(
                recall,
                deltas,
                k_vector=[k - 1 for k in k_vector],
                optim_kwargs=optim_kwargs,
                verbose=verbose,
                guess=guess,
            )
            # reject solution if not within bounds. Normally using L-BFGS-B but here BFGS just for the hess_inv computation to compare.
            if (
                inference_results.x[0] <= 1e-5
                or inference_results.x[0] >= 0.1
                or inference_results.x[1] <= 0
                or inference_results.x[1] >= 1
            ):
                result[..., i] = numpy.nan
                continue

            J = ef_observed_information_matrix(
                recall, deltas, *inference_results.x, k_vector=[k - 1 for k in k_vector]
            )
            try:
                covar = numpy.linalg.inv(J)
            except numpy.linalg.LinAlgError:
                result[..., i] = numpy.nan
                result[:, 0, i] = inference_results.x
                continue
            hess_inv = inference_results.hess_inv
            # test coverage and plot
            cis = CI_asymptotical(covar, inference_results.x, critical_value=1.96)
            cis_hess_inv = CI_asymptotical(
                hess_inv, inference_results.x, critical_value=1.96
            )

            transformed_covar = covar_delta_method_log_alpha(
                inference_results.x[0], covar
            )
            x = [numpy.log10(inference_results.x[0]), inference_results.x[1]]
            cis_log = CI_asymptotical(transformed_covar, x, critical_value=1.96)

            result[:, 0, i] = inference_results.x
            TRUE_VALUES = [ALPHA_TRUE, BETA_TRUE]
            TRANSFORM_TRUE_VALUES = [numpy.log10(ALPHA_TRUE), BETA_TRUE]
            for n in range(2):

                if cis[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                    result[n, 1, i] = True
                else:
                    result[n, 1, i] = False
                if cis_hess_inv[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                    result[n, 2, i] = True
                else:
                    result[n, 2, i] = False
                if (
                    cis_log[n][0] <= TRANSFORM_TRUE_VALUES[n]
                    and TRANSFORM_TRUE_VALUES[n] <= cis_log[n][1]
                ):
                    result[n, 3, i] = True
                else:
                    result[n, 3, i] = False

        results[str(N)] = result
    with open("save_data/ef_asymptotic.json", "w") as _file:
        results = {key: value.tolist() for key, value in results.items()}
        json.dump(results, _file)
    results = {key: numpy.array(value) for key, value in results.items()}
else:
    with open("save_data/ef_asymptotic.json", "r") as _file:
        results = json.load(_file)
        results = {key: numpy.array(value) for key, value in results.items()}

print(numpy.nanmean(results["100"], axis=2))

xaxis = [float(k) for k in results.keys()]
coverage_array = numpy.zeros((2, 2, len(xaxis)))
for n, (key, value) in enumerate(results.items()):
    coverage_array[0, :, n] = numpy.mean(value[0, :, :], axis=1)
    coverage_array[1, :, n] = numpy.mean(value[1, :, :], axis=1)

import seaborn as sb
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.suptitle("Coverage of computed confidence intervals")

sb.lineplot(
    x=xaxis, y=coverage_array[0, 0, :], ax=axs[0], label=r"$\alpha~\mathrm{coverage}$"
)
ax = sb.lineplot(
    x=xaxis, y=coverage_array[0, 1, :], ax=axs[0], label=r"$\beta~\mathrm{coverage}$"
)
ax.set(xscale="log")
ax.set_xlabel("Sample size")
ax.set_ylabel("Coverage")
ax.set_title(r"$\alpha,~\beta$")

sb.lineplot(
    x=xaxis,
    y=coverage_array[1, 0, :],
    ax=axs[1],
    label=r"$\log_{10}(\alpha)~\mathrm{coverage}$",
)
ax = sb.lineplot(
    x=xaxis, y=coverage_array[1, 1, :], ax=axs[1], label=r"$\beta~\mathrm{coverage}$"
)

ax.set(xscale="log")
ax.set_xlabel("Sample size")
ax.set_ylabel("Coverage")
ax.set_title(r"$\log_{10}(\alpha),~\beta$")
plt.savefig("CIcoverage_log.pdf")
plt.show()
