# WeShap
WeShap is a comprehensive evaluation metric for labeling functions in weak supervision. It is based on the concept of Shapley values and is designed to evaluate the quality of labeling functions in weak supervision. WeShap can be used to rank labeling functions, correct labeling functions, and evaluate the quality of weak supervision.

## Installation
```angular2html
pip install -r requirements.txt
```
## Usages
Please check run_pws.py for the configurations of the experiments.Here we provide some example use cases:

Run experiments for ranking LFs on the Yelp dataset
```
python run_pws.py --dataset-name yelp --feature-extractor sentence-bert --lf-tag ngram_random_100 --mode rank
```

Run experiments for correcting LFs on the Indoor-Outdoor dataset
```
python run_pws.py --dataset-name indoor-outdoor --mode revise --revision-type mute
```
