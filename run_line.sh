#!/usr/bin/env bash

./LINE/linux/reconstruct -train $1.csv -output $1_dense.csv -depth 2 -k-max 1000
./LINE/linux/line -train $1_dense.csv -output $1_vec_1st_wo_norm.csv -binary 1 -size 128 -order 1 -negative 5 -samples 30896 -threads 4
./LINE/linux/line -train $1_dense.csv -output $1_vec_2st_wo_norm.csv -binary 1 -size 128 -order 2 -negative 5 -samples 30896 -threads 4
./LINE/linux/normalize -input $1_vec_1st_wo_norm.csv -binary -output $1_vec_1st.csv -binary 1
./LINE/linux/normalize -input $1_vec_2st_wo_norm.csv -binary -output $1_vec_2nd.csv -binary 1
./LINE/linux/concatenate -input1 $1_vec_1st.csv -input2 $1_vec_2nd.csv -output $1_vec_all.txt -binary 1