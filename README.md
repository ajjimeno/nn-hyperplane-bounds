# nn-hyperplane-bounds


Create an environment, for instance in anaconda do:


```
conda create -n bounds python=3
```

and activate your environment:

```
conda activate bounds
```

You need to install numpy, pytorch==1.10.1 and torchvision==0.11.2. Find more information to install pytorch [here](https://pytorch.org/get-started).

To run the experiments on MNIST and CIFAR, run:


```
python Experiments.py
```

Experiments will run for a large set of combinations of parameters, consider selecting the parameters within the code to run a specific subset of experiments.


## Cite us

```
@inproceedings{jimeno2022bounds,
  title={Hyperplane bounds for neural feature mappings},
  author={Antonio Jimeno Yepes},
  booktitle={arxiv},
  year={2022},
}
```

