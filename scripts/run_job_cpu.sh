#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-24:0:0
#SBATCH -p short

#SBATCH --cpus-per-task=1
#SBATCH --job-name=$1
#SBATCH --mem=1G

#SBATCH -o "/work/vig/hamza/video_understanding/lfav_experiments/logs/%j.out"
#SBATCH -e "/work/vig/hamza/video_understanding/lfav_experiments/logs/%j.err"

source "/shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate lfav
cd /work/vig/hamza/video_understanding/lfav_experiments

if [ -z "$2" ]
then
    srun python $1
else
    srun python $1 $2
fi

exit 0
EOT