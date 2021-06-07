<h1 align="center">
    High Performance Computing: Speeding up DNN training
</h1>
<p align="center">
<b>Habib Slim</b>
</p>

# Report
Click [HERE](report.pdf) to download the final report.

# Dependencies
This project uses the following external dependencies:
- Eigen3, for basic matrix operations
- Open MPI 2.1.1

# Usage
The list of experiments available is as follows:
- `param_avg`: Weight averaging algorithm described in the report.
- `parallel_sgd`: Gradient averaging algorithm described in the report.
- `w_param_avg`: Weighted parameter averaging algorithm described in the report.

To compile and run the experiments, from the root directory:

`make [experiment_name]`

And then, for an MPI experiment:

`mpiexec -n n_cores runmpi -options`

Parameters are as follows. For all experiments, the following arguments are available:
- `-batch_size`: Size of each batch
- `-eval_acc`  : To be set to `1` if validation accuracies must be evaluated, `0` otherwise (in which case epoch durations are logged instead).
- `-n_epochs`  : Total number of epochs

Specifically to the following methods, additional parameters are available

**param_avg, w_param_avg**
- `-avg_freq`: Weight averaging frequency (in epochs).

**w_param_avg**
- `-lambda`: Value of the lambda parameter (integer, divided by 100).


# Main references
1. [Ben-Nun et al., 2018] Demystifying parallel and distributed deep learning: An in-depth concurrency analysis
2. [Ericson et al., 2017] On the Performance of Network Parallel Training in Artificial Neural Networks
