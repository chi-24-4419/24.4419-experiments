import numpy
import matplotlib.pyplot as plt

import json

from imlmlib.ef import ExponentialForgetting
from imlmlib.mem_utils import Schedule, experiment, GaussianPopulation
from imlmlib.information import gen_hessians, compute_full_observed_information


def play_blockbased_schedule(population_model, N):
    trials_per_day = int(N / 5)
    blocks = int(N / trials_per_day)
    block_times = numpy.linspace(0, 2000, trials_per_day)
    times = []
    pause = (10 * 86400 - 2000 * blocks) / (blocks)
    for i in range(blocks):
        times += (block_times + i * (2000 + pause)).tolist()
    items = [0 for i in times]
    schedule_one = Schedule(items, times)
    recall = experiment(population_model, schedule_one).squeeze()[0, :]
    deltas = [numpy.infty] + numpy.diff(times).tolist()
    k_repetition = [-1 + i for i in range(N)]
    return recall, deltas, k_repetition


SEED = 456
N = 50
rng = numpy.random.default_rng(seed=SEED)
REPETITION = 1000

ALPHA_TRUE = 1e-2
BETA_TRUE = 4e-1

SUBSAMPLE = 10
subsample_sequence = numpy.logspace(0.5, numpy.log10(N), int(N / SUBSAMPLE))
GEN_DATA = True


if GEN_DATA:

    population_model = GaussianPopulation(
        ExponentialForgetting,
        mu=numpy.array([1e-2, 4e-1]),
        sigma=1e-6 * numpy.eye(2),
        seed=None,
    )
    play_schedule = play_blockbased_schedule
    play_schedule_args = (N,)

    optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-3, 0.7),
        "verbose": False,
    }
    filename = "save_data/ef_schedule_block-based.json"

    json_data, _ = gen_hessians(
        N,
        REPETITION,
        [ALPHA_TRUE, BETA_TRUE],
        population_model,
        play_schedule,
        subsample_sequence,
        play_schedule_args=play_schedule_args,
        optim_kwargs=optim_kwargs,
        filename=filename,
    )

else:
    with open("save_data/ef_schedule_block-based.json", "r") as _file:
        json_data = json.load(
            _file,
        )


recall_array = numpy.asarray(json_data["recall_array"])
observed_hessians = numpy.asarray(json_data["observed_hessians"])
estimated_parameters = numpy.asarray(json_data["estimated_parameters"])
recall_array = recall_array.transpose(1, 0)

recall_kwargs = {
    "x_bins": 100,
}
observed_information_kwargs = {"x_bins": 100}


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
fischer_information, agg_data, inf, cum_inf = compute_full_observed_information(
    [ALPHA_TRUE, BETA_TRUE],
    recall_array,
    observed_hessians,
    estimated_parameters,
    subsample_sequence,
    axs=axs.ravel(),
    recall_kwargs=recall_kwargs,
    observed_information_kwargs=observed_information_kwargs,
    bias_kwargs=None,
    std_kwargs=None,
)
axs[0, 0].set_xscale("log")
axs[0, 0].tick_params(axis="x", which="minor", bottom=False)
axs[0, 0].set_xticks(
    ticks=[int(ss) for ss in subsample_sequence],
    labels=[str(int(ss)) for ss in subsample_sequence],
)


axs[0, 1].set_xscale("log")
axs[0, 1].set_xticks(
    ticks=[int(ss) for ss in subsample_sequence],
    labels=[str(int(ss)) for ss in subsample_sequence],
)
axs[1, 1].set_ylim([1e-2, 1e-1])

plt.tight_layout(w_pad=2, h_pad=0)
plt.get_current_fig_manager().full_screen_toggle()
plt.savefig("images/block.pdf")
