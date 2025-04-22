# RadioGalaxies-BBB
The dev branch has changed quite a bit over the last few years. Check out the main branch for a minimal working example and to replicate results from the 'Approx Bayesian Inference' [paper] (https://proceedings.mlr.press/v244/mohan24a.html).
This branch contains code to explore different modifications to the BBB algorithm, for example:

- Removing the non-linear parameterisation of the std parameters following Kim et al, NeurIPS 2023 [1].
- Using the iVON optimser [2]
## Training
 

- Run bbb_ensemble.py to train 10 variational inference models with different random seeds and random shuffling between training  and validation datasets 
- Use the config_augment.txt to run vi with data agumentation


[1] [On the Convergence of Black-Box Variational Inference](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8bea36ac39e11ebe49e9eddbd4b8bd3a-Abstract-Conference.html) 

[2] [Variational Learning is Effective for Large Deep Networks](https://proceedings.mlr.press/v235/shen24b.html)