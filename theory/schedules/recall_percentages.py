# from imlmlib.mem_utils import BlockBasedSchedule
# from imlmlib.abc import ef_simulator
# from imlmlib.ef import diagnostics
# import numpy








##############
from imlmlib.mem_utils import experiment, GaussianPopulation, trial
from imlmlib.ef import ExponentialForgetting, diagnostics
import functools
import numpy
import arviz as az
from imlmlib.mem_utils import BlockBasedSchedule


# L1 - R1 -  L2 - R2 - L3 - R3   -   R4 - L4 - R5 
#  [800, 800, 800, 800,  800,  86400,  800, 800]
# interblock_time=[8000, 8000, 8000, 8000,  8000,  100000,  8000, 8000]

# ALPHA = 10**(-4)
# BETA = .5
# SIGMA = ALPHA*1e-5*numpy.array([[1,0], [0,.1]]) # same magnitude variation for both parameters
# repet_trials = 1
# nitems = 15
# pop_size = 24
# replications = 1


# default_population_kwargs = {"mu": [ALPHA, BETA], "sigma": SIGMA, "seed": None, 'n_items': nitems, 'population_size': pop_size}
# # L1 - R1 -  L2 - R2 - L3 - R3   -   R4 - L4 - R5 
# #  [800, 800, 800, 800,  800,  86400,  800, 800]
# schedule = BlockBasedSchedule(nitems, 5, interblock_time, repet_trials=repet_trials)
# # for i,t in schedule:
# #     print(i,t)

# population_model = GaussianPopulation(ExponentialForgetting, **default_population_kwargs, )
# for model in population_model:
#     print(model)
# exit()

# data = experiment(population_model, schedule, replications=replications)
# nblock = len(schedule.interblock_time) + 1

# data = (
#     data[0, 0, ...]
#     .transpose(1, 0)
#     .reshape(
#         population_model.pop_size,
#         nblock,
#         schedule.nitems * schedule.repet_trials,
#     )
# )

# times = numpy.array(schedule.times).reshape(-1, repet_trials*nitems)
# deltas = numpy.zeros(times.shape)
# deltas[1:,:] = numpy.diff(times, axis=0)
# deltas[0,:] = numpy.inf
# blocks = numpy.array(schedule.blocks).reshape(-1, repet_trials*nitems)
# deltas_full = numpy.repeat(deltas[numpy.newaxis,:,:], pop_size, axis = 0)
# blocks_full = numpy.repeat(blocks[numpy.newaxis,:,:], pop_size, axis = 0)
# k_repetition = blocks_full -1
















def simulator_block_ef(ALPHA, BETA, SIGMA, repet_trials, nitems, pop_size, replications, interblock_time):
    if repet_trials != 1:
        raise NotImplementedError

    default_population_kwargs = {"mu": [ALPHA, BETA], "sigma": SIGMA, "seed": None, 'n_items': nitems, 'population_size': pop_size}
# L1 - R1 -  L2 - R2 - L3 - R3   -   R4 - L4 - R5 
#  [800, 800, 800, 800,  800,  86400,  800, 800]
    schedule = BlockBasedSchedule(nitems, 5, interblock_time, repet_trials=repet_trials)


    population_model = GaussianPopulation(ExponentialForgetting, **default_population_kwargs, )
    data = experiment(population_model, schedule, replications=replications)
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

    times = numpy.array(schedule.times).reshape(-1, repet_trials*nitems)
    deltas = numpy.zeros(times.shape)
    deltas[1:,:] = numpy.diff(times, axis=0)
    deltas[0,:] = numpy.inf
    blocks = numpy.array(schedule.blocks).reshape(-1, repet_trials*nitems)
    deltas_full = numpy.repeat(deltas[numpy.newaxis,:,:], pop_size, axis = 0)
    blocks_full = numpy.repeat(blocks[numpy.newaxis,:,:], pop_size, axis = 0)
    k_repetition = blocks_full -1

    #verify with diagnostics



    # plt.style.use(style="fivethirtyeight")
    # # fig, (fg, ax, estim) = diagnostics(ALPHA, BETA, k_repetition[:,1:,:].ravel(), deltas_full[:,1:,:].ravel(), data[:,1:,:].ravel())
    # fig, (fg, ax, estim) = diagnostics(ALPHA, BETA, k_repetition.ravel(), deltas_full.ravel(), data.ravel())
    # ax = fig.axes[0]
    # ax.set_xlim([-10,0.5])
    # plt.tight_layout()
    # plt.show()
    # exit()

    ## get mean and std

    # mean, std, std_prob = data.mean(axis=(0, 2)).squeeze(), data.std(axis=(0, 2)).squeeze(), numpy.std(data.mean(axis=2), axis=0).squeeze()

    # return mean, std_prob, data
    return data, deltas_full, k_repetition



if __name__ == '__main__':

    import seaborn
    import matplotlib.pyplot as plt
    import pandas
    import scipy
    
    from imlmlib.ef import plot_exponent_scatter, identify_ef_from_recall_sequence, ef_observed_information_matrix, covar_delta_method_log_alpha
    from imlmlib.mle_utils import CI_asymptotical, confidence_ellipse


    TTEST = True
    ML = True


    ALPHA = 10**(-2)
    BETA = .75
    SIGMA = ALPHA*1e-5*numpy.array([[1,0], [0,.1]]) # same magnitude variation for both parameters
    repet_trials = 1
    nitems = 15
    pop_size = 24
    replications = 1

    interblock_time=[10, 200, 600, 600,  600,  86400,  600, 600] # À la BodyLoci
    RECALL_BLOCKS = numpy.array([1,3,5,6,8])


    data_1, deltas_1, k_repetition_1 = simulator_block_ef(ALPHA, BETA, SIGMA, repet_trials, nitems, pop_size, replications, interblock_time=interblock_time)

    block_average_mean_1 = numpy.mean(a=data_1, axis = 2)
    mean = numpy.mean(block_average_mean_1, axis = 0)
    std = numpy.std(block_average_mean_1, axis = 0)
    se =std/numpy.sqrt(block_average_mean_1.shape[0])
    mean_display_1, se_display_1 = mean[RECALL_BLOCKS], se[RECALL_BLOCKS]


    ALPHA = 10**(-2)
    BETA=.75
    data_2, deltas_2, k_repetition_2 = simulator_block_ef(ALPHA, BETA, SIGMA, repet_trials, nitems, pop_size, replications, interblock_time)
    block_average_mean_2 = numpy.mean(data_2, axis = 2)
    mean = numpy.mean(block_average_mean_2, axis = 0)
    std = numpy.std(block_average_mean_2, axis = 0)
    se =std/numpy.sqrt(block_average_mean_2.shape[0])
    mean_display_2, se_display_2 = mean[RECALL_BLOCKS], se[RECALL_BLOCKS]

    

    ##

    identifier = numpy.tile(numpy.array(range(9)), (block_average_mean_1.shape[0],1))
    
    mean_1_recall = block_average_mean_1[:,RECALL_BLOCKS]
    identifier_1 = identifier[:,RECALL_BLOCKS]
    mean_2_recall = block_average_mean_2[:,RECALL_BLOCKS]
    identifier_2 = identifier[:,RECALL_BLOCKS]
    df = pandas.DataFrame({"block recall %": numpy.concatenate((mean_1_recall.ravel(), mean_2_recall.ravel()), axis = 0), "block": numpy.concatenate((identifier_1.ravel(), identifier_2.ravel()), axis = 0), "Condition": numpy.concatenate((numpy.full(mean_1_recall.ravel().shape, 'A'), numpy.full(mean_1_recall.ravel().shape, 'B')), axis = 0)})


    # paired t-tests:

    if TTEST:
    
        # can be combined like so, but not independent;
        # https://en.wikipedia.org/wiki/Fisher%27s_method

        pvalues = scipy.stats.ttest_rel(mean_1_recall,mean_2_recall, axis = 0).pvalue

        
        
        plt.style.use(style="fivethirtyeight")

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        seaborn.barplot(data = df, x= 'block', y="block recall %", hue = 'Condition',  errorbar = 'se', ax = ax)
        plt.show()


    # =========== ML treatment

    # PreTest verif with True value
    # ALPHA = 1e-2
    # BETA = .5
    # recall, deltas, krepet = data_1.ravel(), deltas_1.ravel(), k_repetition_1.ravel()

    # exponent = [
    #     -ALPHA * (1 - BETA) ** (k) * dt for (k, dt) in zip(krepet, deltas)
    # ]

    # exponent = numpy.nan_to_num(exponent, neginf = -10)

    # _exponent_kwargs = {"xbins": int(len(deltas) ** (1 / 3))}
    # ax, regplot = plot_exponent_scatter(exponent, recall, ax=None, **_exponent_kwargs)
    # plt.show()

    def ML_plot_CE(data, deltas, krepet, ax, colors=["#B0E0E6", "#87CEEB"]):
        
        recall, deltas, krepet = data[:,RECALL_BLOCKS,:].ravel(), deltas[:,RECALL_BLOCKS,:].ravel(), krepet[:,RECALL_BLOCKS,:].ravel()
        optim_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-5, 0.1), (0, 0.99)]}
        verbose = False
        guess = (1e-3, 0.8) # start with credible guess

        inference_results = identify_ef_from_recall_sequence(
            recall_sequence=recall,
            deltas=deltas,
            k_vector=krepet,
            optim_kwargs=optim_kwargs,
            verbose=verbose,
            guess=guess,
        )
        J = ef_observed_information_matrix(
            recall, deltas, *inference_results.x, k_vector=krepet
        )
        covar = numpy.linalg.inv(J)
        transformed_covar = covar_delta_method_log_alpha(inference_results.x[0], covar)
        x = [numpy.log10(inference_results.x[0]), inference_results.x[1]]
        ax_log = confidence_ellipse(x, transformed_covar, ax=ax, colors=colors)
        ax_log.set_title("CE with alpha log scale")
        return ax, inference_results, transformed_covar

    if ML:

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax, inference_results_A, transformed_covar_A = ML_plot_CE(data_1, deltas_1, k_repetition_1, ax = ax)
        ax, inference_results_B, transformed_covar_B = ML_plot_CE(data_2, deltas_2, k_repetition_2, ax = ax, colors = ['red', 'orange'])
        plt.show()

        # recall, deltas, krepet = data_1.ravel(), deltas_1.ravel(), k_repetition_1.ravel()

        # optim_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-5, 0.1), (0, 0.99)]}
        # verbose = False
        # guess = (1e-3, 0.8) # start with credible guess

        # inference_results = identify_ef_from_recall_sequence(
        #     recall_sequence=recall,
        #     deltas=deltas,
        #     k_vector=krepet,
        #     optim_kwargs=optim_kwargs,
        #     verbose=verbose,
        #     guess=guess,
        # )
        # J = ef_observed_information_matrix(
        #     recall, deltas, *inference_results.x, k_vector=krepet
        # )
        # covar = numpy.linalg.inv(J)
        # transformed_covar = covar_delta_method_log_alpha(inference_results.x[0], covar)
        # x = [numpy.log10(inference_results.x[0]), inference_results.x[1]]
        # cis = CI_asymptotical(transformed_covar, x, critical_value=1.96)
        # ax_log = confidence_ellipse(x, transformed_covar, ax=None)
        # ax_log.set_title("CE with alpha log scale")
        # plt.show()


        

    exit()



















# import warnings
# warnings.filterwarnings('default')

# repet_trials = 2
# nitems = 15
# pop_size = 24

# population_kwargs = {
#         "population_size": pop_size,
#         "n_items": nitems,
#         "mu": [5e-3, .4], 
#         "sigma": 1e-6*numpy.eye(2), 
#         "seed": None
#     }

# # schedule
# schedule = BlockBasedSchedule(nitems, 5, [2000, 2000, 86400, 2000], repet_trials=repet_trials)

# # various reshapes
# times = numpy.array(schedule.times).reshape(-1, repet_trials*nitems)
# blocks = numpy.array(schedule.blocks).reshape(-1, repet_trials*nitems)
# times_full = numpy.repeat(times[numpy.newaxis,:,:], pop_size, axis = 0)
# blocks_full = numpy.repeat(blocks[numpy.newaxis,:,:], pop_size, axis = 0)
# k_repetition = blocks_full-1

# #simulate experiment
# mean, std, data = ef_simulator(schedule,population_kwargs=population_kwargs)
# agg_std = numpy.std( numpy.mean(data, axis = 2)  , axis = 0)

# #verify with diagnostics
# alpha = 5e-3
# beta = .4


# # k_repetition_without_first = 
# fig, (fg, ax, estim) = diagnostics(alpha, beta, k_repetition, deltas, recall)