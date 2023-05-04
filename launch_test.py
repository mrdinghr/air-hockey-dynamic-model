from itertools import product

import torch
from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster


LOCAL = False # bool_local_cluster()
TEST = False
USE_CUDA = False

PARTITION = None  # 'amd', 'rtx', 'test30m' for lichtenberg
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1

CONDA_ENV = 'EKF-puck'  # None

N_SEEDS = 1

if LOCAL:
    JOBLIB_PARALLEL_JOBS = 1  # os.cpu_count()
else:
    JOBLIB_PARALLEL_JOBS = 4

N_CORES_JOB = 1
MEMORY_SINGLE_JOB = 1000

launcher = Launcher(exp_name='EKF_puck',
                    python_file='test',
                    #project_name='project01616',
                    n_exps=N_SEEDS,
                    joblib_n_jobs=JOBLIB_PARALLEL_JOBS,
                    n_cores=JOBLIB_PARALLEL_JOBS * N_CORES_JOB,
                    memory_per_core=MEMORY_SINGLE_JOB,
                    days=1,
                    hours=24,
                    minutes=0,
                    seconds=0,
                    partition=PARTITION,
                    conda_env=CONDA_ENV,
                    gres=GRES,
                    use_timestamp=True
                    )


lr = 1e-5
plot = False
epoch = 400
save_gap = 5
losstyps = ['full_log_like', 'log_like', 'multi_mse', 'vomega_log_like'] # mse log_like  multi_mse  multi_log_like  xyomega_log_like  xyomega_mse vomega_log_like vomega_mse full_log_like full_mse
for losstyp in losstyps:
    launcher.add_experiment(lr=lr, save_gap=save_gap, plot=plot, epoch=epoch, loss_type=losstyp)
launcher.run(LOCAL, TEST)
