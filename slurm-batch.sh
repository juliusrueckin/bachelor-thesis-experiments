#!/bin/bash
#SBATCH --cpus 80 
#SBATCH --mem 100GB 
#SBATCH --job-name blog-catalog-preprocessing-v1 
docker stop julius-experiments-container
docker rm julius-experiments-container
docker build --tag julius-experiments -f Dockerfile . 
docker run -d --name julius-experiments-container julius-experiments
docker exec julius-experiments-container python3 /bachelor-thesis-experiments/preprocessing-blogcatalog.py
docker cp /bachelor-thesis-experiments/data/BlogCatalog-dataset/data/blogcatalog_sampling_v1.p ~/julius/results/
