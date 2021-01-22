#!/bin/bash

#$ -N faridNa
#$ -e error_faridNa.txt
#$ -o output_faridNa.txt
#$ -m abe
###$ -m abe
#$ -M farid.najar@universite-paris-saclay.fr
#$ -cwd
#$ -l h_vmem=32G 

module load anaconda/2020.07

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda/2020_07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda/2020_07/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda/2020_07/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda/2020_07/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate torch

## to submit a job : qsub run.sh

## to show current jobs : qstat
## qstat -f -j <job_id> 

## to remove a job : qdel job_id 

python3 farid.py > log_faridNa.txt 

