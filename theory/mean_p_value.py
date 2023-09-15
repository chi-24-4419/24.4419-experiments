from imlmlib.mem_utils import (
    Schedule,
    experiment,
    GaussianPopulation,
    serialize_experiment,
)
from imlmlib.ef import ExponentialForgetting, diagnostics, identify_ef_from_recall_sequence, ef_observed_information_matrix, covar_delta_method_log_alpha
from imlmlib.mle_utils import confidence_ellipse
from imlmlib.information import gen_hessians,  compute_observed_information

import numpy
from imlmlib.mem_utils import BlockBasedSchedule
import matplotlib.pyplot as plt
import seaborn
import numpy
import pandas
import scipy
from tqdm import tqdm


def simulator_block_ef(ALPHA, BETA, SIGMA, repet_trials, nitems, pop_size, replications, intertrial_time, interblock_time):
    if repet_trials != 1:
        raise NotImplementedError

    default_population_kwargs = {"mu": [ALPHA, BETA], "sigma": SIGMA, "seed": None, 'n_items': nitems, 'population_size': pop_size}
    schedule = BlockBasedSchedule(nitems, intertrial_time, interblock_time, repet_trials=repet_trials)


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

    return data, deltas_full, k_repetition


def diff_eval(ALPHA_C, ALPHA_A, intertrial_time, interblock_time, condition_names, *args):
    data_C, deltas_C, k_repetition_C = simulator_block_ef(ALPHA_C, *args, intertrial_time, interblock_time)
    block_average_mean_C = numpy.mean(data_C, axis = 2)
    
    data_A, deltas_A, k_repetition_A = simulator_block_ef(ALPHA_A, *args, intertrial_time, interblock_time)
    block_average_mean_A = numpy.mean(data_A, axis = 2)

    identifier = numpy.tile(numpy.array(range(9)), (block_average_mean_C.shape[0],1))

    mean_C_recall = block_average_mean_C[:,RECALL_BLOCKS]
    identifier_C = identifier[:,RECALL_BLOCKS]
    mean_A_recall = block_average_mean_A[:,RECALL_BLOCKS]
    identifier_A = identifier[:,RECALL_BLOCKS]
    df = pandas.DataFrame({"block recall %": numpy.concatenate((mean_C_recall.ravel(), mean_A_recall.ravel()), axis = 0), "block": numpy.concatenate((identifier_C.ravel(), identifier_A.ravel()), axis = 0), "Condition": numpy.concatenate((numpy.full(mean_C_recall.ravel().shape, condition_names[0]), numpy.full(mean_A_recall.ravel().shape, condition_names[1])), axis = 0)})
    return df, data_C, deltas_C, k_repetition_C, data_A, deltas_A, k_repetition_A, mean_C_recall, mean_A_recall



############# 


if __name__ == '__main__':

    plt.style.use(style="fivethirtyeight")

    import pickle
    GEN = False

    # Shared parameters
    BETA = .4
    SIGMA = 1e-7*numpy.array([[1,0], [0,.1]]) # same magnitude variation for both parameters
    repet_trials = 1
    nitems = 1
    pop_size = 100
    replications = 1 
    RECALL_BLOCKS = numpy.array([0,1,2,3,4,5,6,7,8])
    ALPHA_A = 10**(-2.1)
    ALPHA_C = 10**(-1.9)
    ALPHA = 10**(-2)
    N = 100
    REPET = 100



    def play_replicated_schedule(population_model, times):
        items = [0 for i in times]
        schedule_one = Schedule(items, times)
        data = experiment(population_model, schedule_one).squeeze()[0, :]
        data = data.transpose(1, 0)
        data, k_vector, deltas = serialize_experiment(data, times)
        return data, deltas, k_vector

    default_population_kwargs = {"mu": [ALPHA, BETA], "sigma": SIGMA, "seed": None, 'n_items': 1, 'population_size': 100}
    population_model = GaussianPopulation(ExponentialForgetting, **default_population_kwargs, )
    play_schedule = play_replicated_schedule

    optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-3, 0.7),
        "verbose": False,
    }
    ## === A vs C

    p_value_container = numpy.zeros((REPET,9,2))
    intertrial_time = 0

    if GEN:
        for ni,i in enumerate(tqdm(range(1,10))):
            interblock_time = [-numpy.log(i/10)/(ALPHA*(1-BETA)**(k)) for k in range(9)]
            for r in range(REPET):

                play_schedule_args = (interblock_time,)
                df1, data_C1, deltas_C1, k_repetition_C1, data_A1, deltas_A1, k_repetition_A1, mean_C1_recall, mean_A1_recall = diff_eval(ALPHA_C,ALPHA_A, intertrial_time,interblock_time, ['C', 'A'] ,BETA, SIGMA, repet_trials, nitems, pop_size, replications)

            

            ## ==== classical comparison
                
                pvalues = scipy.stats.ttest_rel(mean_A1_recall,mean_C1_recall, axis = 0).pvalue[1:]
                # fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10,7.5))
                # seaborn.barplot(data = df1, x= 'block', y="block recall %", hue = 'Condition',  errorbar = 'se', ax = axs[0])
                # plt.show()
                p_value_container[r,ni,0] = scipy.stats.combine_pvalues(pvalues, method='fisher').pvalue
                p_value_container[r,ni,1] = scipy.stats.combine_pvalues(pvalues, method='stouffer').pvalue
                
            

    #         json_data = gen_hessians(
    #             N,
    #             REPETITION,
    #             [ALPHA_A, BETA],
    #             population_model,
    #             play_schedule,
    #             subsample_sequence,
    #             play_schedule_args=play_schedule_args,
    #             optim_kwargs=optim_kwargs,
    #             filename=None,
    #         )

    #         observed_information_kwargs = {"x_bins": 10, "cumulative": True, "cum_color": "orange"}


    #         observed_hessians = numpy.asarray(json_data["observed_hessians"])
    #         mean_observed_information, information, cum_inf = compute_observed_information(
    #             observed_hessians,
    #             axs=None,
    #             observed_information_kwargs=observed_information_kwargs,
    #             )
    #         container.append(cum_inf)

        with open('data.pkl', 'wb') as _file:
            pickle.dump(p_value_container,  _file)

    else:
        with open('data.pkl', 'rb') as _file:
            p_value_container = pickle.load(_file)


    fig, axs = plt.subplots(nrows=1,ncols=1)
    labels = ['p=0.1', 'p=0.2', 'p=0.3', 'p=0.4', 'p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9']
    ps = [i/10 for i in list(range(1,10))]
    p_values = numpy.mean(p_value_container, axis = 0)
    axs.plot(ps, p_values[:,0], 'o', label = 'fisher p-values')
    axs.plot(ps, p_values[:,1], 'D', label = 'stouffer p-values')
    axs.set_yscale('log')
    axs.legend()
    axs.set_xlabel('schedule aimed probability')
    axs.set_ylabel('combined p-values')
    plt.tight_layout(w_pad=-2, h_pad=0)
    plt.savefig("p_values.pdf")
    plt.show()

