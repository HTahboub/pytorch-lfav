#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-8:0:0
#SBATCH -p gpu

#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=$1
#SBATCH --mem=32G
#SBATCH --open-mode=append

#SBATCH -o "/home/tahboub.h/dynamic-patch/logs/%j.out"
#SBATCH -e "/home/tahboub.h/dynamic-patch/logs/%j.err"
module load cuda/11.7

cd "/scratch/tahboub.h"
source "/shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate patch

if [ -z "$2" ]
then
    srun python $1
else
    srun python $1 $2
fi

exit 0
EOT