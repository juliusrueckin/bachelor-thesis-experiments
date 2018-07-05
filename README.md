# Bachelor Thesis Experiments
## Similarity Explanation and Exploration in a heterogeneous Information Network

### Usage of Competitor's Implementations
#### [VERSE](https://github.com/xgfs/verse)
```shell
cd verse/src && make;
./verse -input data/karate.bcsr -output karate.bin -dim 128 -alpha 0.85 -threads 4 -nsamples 3
```
#### [Deepwalk](https://github.com/xgfs/deepwalk-c)
```shell
cd deepwalk-c/src && make;
./deepwalk -input ../../verse/data/karate.bcsr -output karate.bin -dim 128 -threads 4
```

#### [Node2Vec](https://github.com/xgfs/node2vec-c)
```shell
cd node2vec-c/ && make;
./node2vec -input ../verse/data/karate.bcsr -output karate.bin -dim 128 -threads 4
```
#### [LINE](https://github.com/tangjianpku/LINE)
Download latest GNU GSL Package [here](http://www.singleboersen.com/mirror/gnu/gsl/) and extract it on your local machine. Then do
```shell
cd /gslu-extracted-path && ./configure && make && make install;
cd /path-to-line-directory/linux && make;
./line -train network_file -output embedding_file -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20
```
