# RadioGalaxies-BBB
Code for variational inference for radio galaxy classification based on the Bayes by Backprop algorithm.

## Training
 

- Run bbb_ensemble.py to train 10 variational inference models with different random seeds and random shuffling between training  and validation dataset. Uses wandb logger.

Configs for experiments:
- Use configt.txt to run vi without data agumentation
- Use config_augment.txt to run vi with data agumentation

## Evaluation

- Run pred_analysis.py to calculate test error and uncertainty calibration error for predictive entropy for 10 experimental runs.
- Run analysis.py to calculate energy scores for different test sets (MiraBest, MIGHTEE, GalaxyMNIST).