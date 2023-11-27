#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1-0:0:0
#SBATCH -p jiang

#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=$1
#SBATCH --mem=64G
#SBATCH --open-mode=append

#SBATCH -o "/work/vig/hamza/video_understanding/lfav-experiments/logs/%j.out"
#SBATCH -e "/work/vig/hamza/video_understanding/lfav-experiments/logs/%j.err"
module load cuda/12.1

cd "/work/vig/hamza/video_understanding/lfav-experiments/src"
source "/shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate lfav

if [ -z "$2" ]
then
    srun python $1
else
    srun python $1 $2
fi

exit 0
EOT