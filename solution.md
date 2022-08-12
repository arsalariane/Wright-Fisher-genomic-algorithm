
# Wright-Fisher model
## Libraries

```python
### 1
%matplotlib inline 
#If you run into errors with %matplotlib, check that your version of ipython is >=1.0 
import numpy as np #numpy defines useful functions for manipulating arrays and matrices.
import matplotlib.pyplot as plt #matplotlib is plotting library
```

## Implementation

```python
### 2
p0   = 0.1  # initial proportion of "1" alleles 
nInd = 100  # initial population size (number of individuals)
```

```python
### 3
# Initialize a population of length nInd with only 0 alleles. 
initial_population = np.zeros(nInd)

# Set the first p0*nInd alleles to 1. 

initial_population[0:int(p0*nInd)] = 1

#The position of individuals doesn't matter in this model, but if you prefer to have a more realistically random 
# distribution of alleles, you can use np.random.shuffle to distribute alleles randomly.

np.random.shuffle(initial_population)
```

```python
### 4
sample_size = 10
np.random.choice(initial_population, sample_size, replace=False )
```

*Optional Mathematical exercise*

1. What is the distribution of offspring number per individual in the Wright-Fisher model? 
Answer :   #alternate-alleles, (x) = h(x; nInd, s, p*nInd)
    
    It is a hypergeometric probability distribution. The probability that an s trail hypergeometric experiment results in exactly x alternate alleles, when the population size is nInd, (p*nInd) of which are classified as alternate alleles. The s trails are dependent as they are done without replacement.
    
    h(x; nInd, s, p*nInd) = [ C(p*nInd, x) * C(nInd(1-p), s-x) ] / [ C(nInd, s) ]

2. Convince yourself that this distribution is approximately Poisson distributed with mean one (hint: This is a consequence of the law of rare events)


```python
### 5
import scipy
from scipy import stats

iterations = 10000  # the number of times to draw.
sample_size = 50  # the size of each sample
alt_counts = []  # number of alternate alleles (i.e., 1's) for each draw

for i in range(iterations):
    sample=np.random.choice(initial_population, sample_size, replace=False)
    # get the number of alt alleles
    alt_counts.append(sample.sum())
    
# plot a histogram of sampled values    
plt.hist(alt_counts, sample_size + 1, range=(-0.5, sample_size + 1 - 0.5), label="random sample")
plt.xlabel("number of alt alleles")
plt.ylabel("counts")

# Compare this to some discrete distributions
x_range = range(sample_size + 1) # all the possible values

p = np.sum(initial_population) * 1. / len(initial_population)  # initial fraction of alt's

# poisson with mean sample_size * p
y_poisson = stats.poisson.pmf(x_range, sample_size*p) * iterations
# binomial with probability p and  sample_size draws
y_binom = stats.binom.pmf(x_range, sample_size,p) * iterations
# hypergeometric draw of sample_size from population of size len(initial_populationpop)
# with np.sum(initial_population) ones.
y_hypergeom = stats.hypergeom.pmf(x_range, len(initial_population), np.sum(initial_population), sample_size)\
                * iterations

plt.plot(x_range, y_poisson, label="Poisson", lw=3)
plt.plot(x_range, y_binom, label="Binomial")
plt.plot(x_range, y_hypergeom, label="Hypergeometric")
plt.xlim(-0.5, sample_size + 0.5)
plt.legend()
```

