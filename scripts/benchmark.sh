#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH --gpus 1 # gpus
#SBATCH -o logs/%x.%N.%j.outputs # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%x.%N.%j.errors # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J benchmark_pipeline # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Check GPU allocation
echo "Running Benchmark pipeline..."
# python -m src.qtt.objective.cv.sam2_pipeline.finetune_wrapper
python -m benchmarks.gluon_tests

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";