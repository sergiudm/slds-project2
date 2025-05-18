#!bin/bash

mlx_lm.lora \
    --model /Users/sergiu/.cache/huggingface/hub/models--google--gemma-3-4b-it \
    --train \
    --data /Users/sergiu/ws/slds-project2/assets/douban_movie.csv \
    --iters 600