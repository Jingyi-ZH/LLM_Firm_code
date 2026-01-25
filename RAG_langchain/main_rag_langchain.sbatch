#!/bin/bash
#---------------------------------------------------------------------------------
# Account information
#SBATCH --account=phd
#---------------------------------------------------------------------------------
# Resources requested
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH --ntasks=3                 
#SBATCH --cpus-per-task=1          
#SBATCH --mem-per-cpu=16G          
#SBATCH --time=1-00:10:00          

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)
#SBATCH --job-name=exp_main_batch

#---------------------------------------------------------------------------------
# Print some useful variables
# set -x

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"
start="$(date +"%T")"
echo "Start time : $start"

#---------------------------------------------------------------------------------
# Run the job

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun -n1 -c1 python -V

echo "Begin running: /home/yliu420/Project/Belief-Elicitation-on-LLM-postrelease/src/run_all_fixreal.sbatch"
echo "...python script begin..."

# Launch 5 tasks in parallel (within the same sbatch/salloc allocation)
srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.1.out \
  python main_rag_langchain.py --real_profile_id 'iPhone 17' --account OPENAI_JZ3533 &

# srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.2.out \
#   python main_rag_langchain.py --real_profile_id 'iPhone 17 Air' --account OPENAI_JZhang34Booth &

srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.3.out \
  python main_rag_langchain.py --real_profile_id 'iPhone 17 Pro' --account OPENAI_MLIU &

srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.4.out \
  python main_rag_langchain.py --real_profile_id 'iPhone 17 Pro Max' --account OPENAI_YLIU &

# srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.5.out \
#   python main_rag_langchain.py --real_profile_id 'iPhone 17' --account OPENAI_CU &

# srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.6.out \
#   python main_rag_langchain.py --real_profile_id 'iPhone 17 Air' --account OPENAI_JZ3533 &

# srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.7.out \
#   python main_rag_langchain.py --real_profile_id 'iPhone 17 Pro' --account OPENAI_MLIU &

# srun -n1 -c1 --exclusive -o slurm-%j.step-%s.t-%t.8.out \
#   python main_rag_langchain.py --real_profile_id 'iPhone 17 Pro Max' --account OPENAI_JZhang34Booth &

# Wait for all tasks to finish
wait

echo "All processes completed."
echo "...python script end..."
end="$(date +"%T")"
echo "End time : $end"
