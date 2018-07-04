# Bachelor Thesis Experiments
## Similarity Explanation and Exploration in a heterogeneous Information Network

### Usage of Competitor's Implementations
#### VERSE
```shell
cd verse/src && make;
./verse -input data/karate.bcsr -output karate.bin -dim 128 -alpha 0.85 -threads 4 -nsamples 3
```
#### Deepwalk
```shell
cd deepwalk-c/src && make;
./deepwalk -input ../../verse/data/karate.bcsr -output karate.bin -dim 128 -threads 4
```

#### Node2Vec
```shell
cd node2vec-c/ && make;
./node2vec -input ../verse/data/karate.bcsr -output karate.bin -dim 128 -threads 4
```
