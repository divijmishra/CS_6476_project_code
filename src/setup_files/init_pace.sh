#!/usr/bin/bash

# To use a different GPU: change "V100-32GB" to the appropriate name as per
# https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096 
# To change the time required, change t01:00:00 to tHH:MM:SS

# can also run the bash file with arguments to change GPU and number of hours, e.g.
# bash init_pace.sh -g V100-16GB -h 2    (this change GPU to V100-16GB and requests access for 2 hours)

gpu=A100-80GB
hours=4

while getopts ":g:h:" opt; do
    case "${opt}" in
        g) gpu=$OPTARG;;
        h) hours=$OPTARG;;
    esac
done

# srun --partition=coc-gpu --gres=gpu:1 -C $gpu -N 1 -t $hours:00:00 --pty $SHELL
srun --partition=coc-gpu --mem-per-cpu=32G --gres=gpu:1 -C $gpu -N 1 -t $hours:00:00 --pty $SHELL
