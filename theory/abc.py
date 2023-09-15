from imlmlib.mem_utils import BlockBasedSchedule
from imlmlib.abc import ef_infer_abc, simulator_block_ef, plot_ihd_contours
import numpy

import warnings
warnings.filterwarnings('default')


simulation_kwargs = {
        "pop_size": 24,
        "nitems": 15,
        "seed": None,
        "SIGMA": 1e-7*numpy.eye(2),
        'repet_trials': 1,
        'intertrial_time' : 20,
        'interblock_time': [30, 30, 30, 30, 30, 86400, 30, 30],
        'RECALL_BLOCKS': [1,3,5,6,8],
        "replications": 1
    }


mean_recall = simulator_block_ef(
            simulation_kwargs['SIGMA'],
            simulation_kwargs['repet_trials'],
            simulation_kwargs['nitems'],
            simulation_kwargs['pop_size'],
            simulation_kwargs['replications'],
            simulation_kwargs['intertrial_time'],
            simulation_kwargs['interblock_time'],
            simulation_kwargs['RECALL_BLOCKS'],
            None,
            10**(-2),
            0.7)

    


simulator_kwargs ={"epsilon": .01}
# observed_data = numpy.array([0.30, 0.60, 0.73, 0.53, 0.77])
idata = ef_infer_abc(simulation_kwargs, mean_recall, simulator_kwargs)

import arviz as az
import matplotlib.pyplot as plt


print(az.summary(idata, kind="stats"))
az.plot_forest(idata, var_names=["log10alpha", 'b'], combined=True, hdi_prob=0.95)

ax = plot_ihd_contours(idata)
plt.show()
