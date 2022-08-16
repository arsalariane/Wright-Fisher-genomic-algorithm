
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

What is the distribution of the number of alternate alleles in a sample of s  individuals from a population of size nInd with allele frequency p?
```diff
@@ Answer :  If we are sampling s individuals from a population we cannot choose the same person twice. Also the population size is fixed and it has a fixed number of individual with alternate allele. This is equivalent to “probability of k successes in s draws without replacement from a finite population of size nInd that contains p*nInd objects with that feature”. So this is a hypergeometric probability distribution. The probability that an s trail hypergeometric experiment results in exactly x alternate alleles, when the population size is nInd, (p*nInd) of which are classified as alternate alleles is: h(x; nInd, s, p*nInd) = [ C(p*nInd, x) * C(nInd(1-p), s-x) ] / [ C(nInd, s) ] @@
```

2. Convince yourself that this distribution is approximately Poisson distributed with mean one (hint: This is a consequence of the law of rare events)


```python
### 5
import scipy
from scipy import stats

iterations = 10000  # the number of times to draw.
sample_size = 10 # the size of each sample
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

If we sample 10 individuals from the initial population repeatedly :
#![png](image1.PNG)

If we sample 50 individuals from the initial population repeatedly :
![png](image1bis.PNG)

```diff
@@ It is clear from the simulation as shown above that the number of alternate alleles if we sample 50 individuals from the initial population repeatedly the number of individuals with alternate allele follows hypergeometric distribution.@@
```

```python
### 6
def generation(pop):
    #number of individuals 
    nInd = len(pop)
    #generate the descendant population
    return np.random.choice(pop, nInd, replace=True) 
                    
generation(initial_population)
```

Here again, we get a different number of ones every time we run the "generation" function. I also generated a bunch of generation samples to get an idea of how much variation there is, and overlaid some plausible distribution. Which one fits best? Does it make sense to you?

```diff
@@ Answer :Binomial distribution and Poisson distributions fit quite well. And this is natural, because the sampling is now   done with replacement, meaning independent trials, which are the characteristics of Binomial and Poisson distributions.  Of these two Binomial fits the generated distribution the best. Hypergeometric distribution underfits our distribution   as trails are considered done without replacement in Hypergeometric distributions.@@
```


```python
### 7

nsample = 10000  # the number of samples to draw.
alt_counts = []  # number of alternate alleles (i.e., 1's) for each draw

for i in range(nsample):
    offspring = generation(initial_population)
    alt_counts.append(offspring.sum())

hist = plt.hist(alt_counts, len(initial_population)+1, range=(0-0.5, len(initial_population)+0.5))
plt.xlabel("number of alt alleles")
plt.ylabel("counts")

#Here I just check that the initial population is still a list of length nInd
assert nInd==len(initial_population),"initial_population doesn't have the same length as nInd" 

x_range=range(nInd+1)                  #all the possible values
p=np.sum(initial_population)*1./nInd   #the initial frequency

#Compare this to some distributions
y_poisson=stats.poisson.pmf(x_range, nInd*p) * nsample
y_binom=stats.binom.pmf(x_range, nInd, p) * nsample
y_hypergeom=stats.hypergeom.pmf(x_range, nInd, np.sum(initial_population), nInd) * nsample

plt.plot(x_range, y_poisson, label="Poisson",lw=3)
plt.plot(x_range, y_binom, label="Binomial")
plt.plot(x_range, y_hypergeom, label="Hypergeometric")
plt.xlim(-0.5, nInd+0.5)
plt.ylim(0, 1.2*max(hist[0]))
plt.legend()
```

![png](image2.PNG)

```diff
@@ Here we see the distribution of number of alternate alleles in offsprings if they are allowed to choose a random parent. Since one parent can have multiple offsprings this is a sampling with replacement. So this is equivalent to independent 10000 Bernoulli trials. So this is a binomial distribution.@@
```

Now we are ready to evolve our population for 100 generations. Let's store the entire genotypes for each generation in a list.

```python
### 8
nGen = 100  # number of generations to simulate
history = [initial_population]  # a container list for our simulations. It will contain the population 
                                        # state after generations 0 to nGen
for i in range(nGen):
    # evolve the population for one generation, and append the result to history.
    history.append(generation(history[i])) 
history = np.array(history)  # convert the list into an array for convenient manipulation later on
```
Now we want to look at the results. Let's compute the allele frequency at each generation and plot that as a function of time.

```python
### 9
#compute the allele frequency at each generation.

freqs = np.mean(history, axis=1)
plt.plot(freqs)
plt.axis([0, 100, 0, 1]);#define the plotting range
plt.xlabel("generation")
plt.ylabel("population frequency")
```

![png](image3.PNG)

```diff
@@ As we can see above the allele has gone extinct after 32-33 generations in the particular simulation.@@
```

Now we would like to experiment a bit with the tools that we have developed. Before we do this, we will organize them a bit better, using a Python "class" and object-oriented programming. We have defined above variables that describe a population (such as the population size nInd, and the ancestral frequency g0). We have also defined functions that apply to a population, such as "generation". A class is used to keep track of the relation between objects, variables, and functions.


```python
### 10
class population:
    """
    Initialization call: 
    
    population(nInd,p0)
    requires a number of individuals nInd and an initial frequency p0
    
    Variables:
    nInd: The number of individuals
    p0: the initial allele frequency
    initial_population: an array of nInd alleles
    history: a list of genotypes for each generation 
    frequency_trajectory: an allele frequency trajectory; only defined if get_frequency_trajectory is run. 
    Methods:
    generation: returns the offspring from the current population, whish is also the last one in self.history
    evolve: evolves the population for a fixed number of generations, stores results to self.history
    get_frequency_trajectory: calculates the allele frequency history for the population
    plot_trajectory: plots the allele frequency history for the population
    
    """
    def __init__(self, nInd, p0): 
        """initialize the population. nInd is the number of individuals. p0 is the initial allele frequency.
        __init__ is a method that, when run, creates a "population" class and defines some of its variables. 
        Here we define this __init__ method but we don't run it, so there is no "population" created yet.  
        In the meantime, we'll refer to the eventual population object as "self".
        We'll eventually create a population by stating something like   
        pop = population(nInd,p0)
        This will call the __init__ function and pass a "population" object to it in lieu of self. 
        """
        #Pass nInd & p0 to nInd & p0 in the population object
        self.nInd = nInd
        self.p0 = p0 
        # Initialize the population
        self.initial_population = np.zeros(self.nInd) 
        if (p0 != 0 and int(p0*self.nInd) == 0):
            print("warning: the initial frequency was smaller than 1/nInd. Initial frequency set to zero. ")
        self.initial_population[0 : int(p0*self.nInd)] = 1
        np.random.shuffle(self.initial_population)
        # History is a container that records the genotype at each generation.
        # We'll update this list as we go along. For now it should just contains the initial population
        self.history = [self.initial_population]

    def generation(self): 
        """class methods need "self" as an argument in they definition to know that they apply to a "population" 
        object. If we have a population "pop", we can write 
        pop.generation(), and python will know how to pass the population as the first argument. 
        
        Returns a descendant population according to Wright-Fisher dynamics with constant size, assuming that the
        parental population is the last generation of self.history. 
        """
        return np.random.choice(np.array(self.history[-1]), self.nInd, replace=True) 

    def evolve(self,nGen): 
        """
        This is a method with one additional argument, the number of generations nGen. 
        To call this method on a population "pop", we'd call pop.evolve(nGen). 
        This function can be called many times on the same population. 
        pop.evolve(2)
        pop.evolve(3)
        would evolve the population for 5 generations. 
        For each step, we make a call to the function generation() and append the population to the "self.history"
        container. 
        """
        for _i in range(nGen):
            self.history.append(self.generation())
        self.get_frequency_trajectory() # This computes self.frequency_trajectory, the list of population frequencies from the first generation 
                       # to the current one. the get_frequency_trajectory function is defined below. 

    def get_frequency_trajectory(self):
        """
        returns an array containing the allele frequency history for the population (i.e., the allele frequency at 
        generations 0,1,2,..., len(self.history))
        """
        history_array = np.array(self.history) 
        self.frequency_trajectory = np.mean(history_array, axis=1)   
        return self.frequency_trajectory

    def plot_trajectory(self,ax="auto"):
        """
        plots the allele frequency history for the population
        """
        
        plt.plot(self.frequency_trajectory)
        if ax=="auto":
            plt.axis([0, len(self.history), 0, 1]) 
        else:
            plt.axis(ax)
    

        

            
        
        
```

# Exploration
## Drift
We can now define multiple populations, and let them evolve from the same initial conditions.


```python
### 11
nInd = 50
nGen = 30
nRuns = 30
p0 = 0.02
# Create a list of length nRuns containing initial populations 
# with initial frequency p0 and nInd individuals.
pops = [population(nInd, p0) for i in range(nRuns)] 
```

Evolve each population for nGen generations. Because each population object has it's own internal storage for the history of the population, we don't have to worry about recording anything.


```python
### 12
for pop in pops:
    pop.evolve(nGen);  
```

Now plot each population trajectory, using the built-in method from the population class. 


```python
### 13
for pop in pops:
    pop.plot_trajectory();
plt.xlabel("generation")
plt.ylabel("population frequency of 1 allele") 
```

![png](image4.PNG)

```diff
@@We can see the drift in 10 independent populations over 30 generations starting with allele frequency 0.02 (which corresponds to p0)@@
```

Now that we know it works, let's explore this a bit numerically. Try to get at least 1000 runs, it'll make graphs prettier down the road.


```python
### 14
nInd = 100
nGen = 30 
nRuns = 2000 
p0 = 0.02
pops = [population(nInd, p0) for i in range(nRuns)] 
for pop in pops:
    pop.evolve(nGen);
    pop.plotTraj();

plt.xlabel("generation")
plt.ylabel("population frequency") 

```

![png](image5.PNG)

So there is a lot of randomness in there, but if you run it multiple times you should see that there is some regularity in how fast the allele frequencies depart from the initial values.
To investigate this, we will generate a histogram describing the distribution of allele frequencies at each generation.

```python
### 15
def frequencyAtGen(generation_number, populations, nBins=11):
    """calculates the allele frequency at generation genN for a list of populations pops. 
     Generates a histogram of the observed values"""
    # First build a list of frequencies observed at generation generation_number in 
    # the different populations. WE will then build a histogram from this list
    list_of_frequencies = [pop.frequency_trajectory[generation_number]
                                          for pop in populations] 
    counts_per_bin, bin_edge_positions = np.histogram(list_of_frequencies, 
                                                      bins=nBins, range=(0,1)) 
    
    bin_centers=np.array([(bin_edge_positions[i+1]+bin_edge_positions[i]) / 2 for i in range(len(counts_per_bin))]) 
    return bin_centers, counts_per_bin # Return the data from which we will generate the plot
    
```


```python
### 16

nBins = 11 # The number of frequency bins that we will use to partition the data.
for i in range(nGen+1):
    
    bin_centers, counts_per_bin = frequencyAtGen(i, pops, nBins); 
    if i==0:
        plt.plot(bin_centers, counts_per_bin, color=plt.cm.autumn(i*1./nGen), label="first generation")  
                                                                        # cm.autumn(i*1./nGen) returns the 
                                                                        # color with which to plot the current line
    elif i==nGen:
        plt.plot(bin_centers, counts_per_bin, color=plt.cm.autumn(i*1./nGen), label="generation %d"% (nGen,))
    else:
        plt.plot(bin_centers, counts_per_bin, color=plt.cm.autumn(i*1./nGen))
plt.legend()
plt.xlabel("Population frequency")
plt.ylabel("Number of simulated populations ")
```


![png](image6.PNG)


There are three important observations here:

    1-Frequencies tend to spread out over time 
    2-Over time, there are more and more populations at frequencies 0 and 1. (Why?) 
```diff
@@ Answer: We can observe a saturation taking place at 0 and 1 : some population reach extinction of the allele and it stays at 0 and, but if fixation is achieved it stays at 1. Even after further generations those populations remain in the same category. But other populations which were in between can reach these saturation in subsequent generations. @@
```

    3-Apart from the 0 and 1 bins, the distribution becomes entirely flat.

A few alternate ways of visualizing the data: first a density map


```python
### 17

nBins = 11
sfs_by_generation = np.array([frequencyAtGen(i, pops, nBins=nBins)[1] for i in range(0, nGen+1)])
bins = frequencyAtGen(i, pops, nBins=nBins)[0]
plt.imshow(sfs_by_generation, aspect=nBins*1./nGen, interpolation='nearest')
plt.xlabel("Population frequency (bin number)")
plt.ylabel("Generation")
plt.colorbar()
```
for p0 = 0.02
![png](image7.PNG)

for p0 = 0.3
![png](image7bis.PNG)

```diff
@@ As we can see all 2000 populations were in 3rd bin initially and spreads out in subsequent generations.@@
``` 

Then a 3D histogram, unfortunately a bit slow to compute. 


```python
### 18
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d', elev=90)
xedges = bins
yedges = np.arange(nGen+1)

xpos, ypos = np.meshgrid(xedges-.4/nBins, yedges-0.5)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = 0 * ypos
dx = .8 / nBins
dy = 1
dz = sfs_by_generation.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', edgecolor='none', alpha=0.15)

ax.view_init(elev=30., azim=60)
ax.set_xlabel("Population frequency")
ax.set_ylabel("Generation")
ax.set_zlabel("Counts")

plt.show()
```

For p0 =0.3

![png](image8bis.PNG)


Now let's dig into the effect of population size in a bit more detail. Consider the change in the frequency of alleles between parent and offspring.

*Mathematical exercise (NOT optional)*:

1. What is the expected distribution of allele frequencies after one generation, if they start at frequency $p$ in a population of size $N$? 
 
```diff
@@ Answer :The expected distribution of allele frequencies after one generation is binomial. It is mathematically defined as follows: Frequency of alternate alleles after one generation, j = Binomial( $N$, $p$) = C(N,j) * p^j (1-p)^(N-j) @@
```

2. What is the variance of this distribution? (Look it up if you don't know--wikipedia is useful for that kind of stuff)

```diff
@@ Answer :  Variance = Np(1-p) @@
```

3. What is the variance in the distribution in the derived allele frequency (rather than the allele counts)?

```diff
@@ Answer :  Variance = p(1-p)/N @@
```
To study the effect of population size on the rate of change in allele frequencies, plot the distribution of allele frequencies after nGen generation. Start with nGen=1 generation.



```python
### 19
histograms = []
variances = []
p0 = 0.2
sizes = [5, 10, 20, 50, 100, 500] 
nGen = 1
for nInd in sizes:
    pops = [population(nInd, p0) for i in range(1000)] 
    [pop.evolve(nGen) for pop in pops]
    sample = [pop.get_frequency_trajectory()[-1] for pop in pops]
    variances.append(np.var(sample))
    histograms.append(plt.hist(sample, alpha=0.5, label="size %d" % (nInd,) ))
plt.xlabel("Population frequency")
plt.ylabel("Number of populations")
plt.legend()
```

![png](image9.PNG)


So how does population size affect the change in allele frequency after one generation? Can you give a specific function describing the relationship between variance and population size?

```diff
@@ Answer : The variance in allele frequency is decreasing with increase in population size. When population size decreases the distribution spreads out i.e variance increases. This is in accordance with the theory we developed above. Variance = 1/N. This function is more and more accurate for large values of N. @@
```

You can get this relationship from the math exercise above, or just try to guess it from the data. If you want to try to guess, start by plotting the variances (stored in "variances") against the population sizes (stored in "sizes"). Then you can either try to plot different functinoal forms to see if they fit, or you can change the way you plot the data such that it looks like a straight line. If you do the latter, make sure you update the labels!

Here I'm giving you a bit more room to explore--there are multiple ways to get there.  

```python
### 20
plt.plot(np.array(sizes), variances, 'o', label="simulation") #this is a starting point, but you can change this!
# Your theory.
my_variance = [1/x for x in sizes]
plt.plot(np.array(sizes), np.array(my_variance), 'x', label='theory')

plt.xlabel("Population size") 
plt.ylabel("Variance")  
```


![png](image10.PNG)


For short times, the expected changes in allele frequencies, $Var\left[E[(x-x_0)^2)\right]$, are larger for smaller population, a crucial result of population genetics. 

The next question is: How does the rate of change in allele frequency depend on the initial allele frequency? We can plot the histograms of allele frequency as before:


```python
### 21
histograms = []
variances = []
p0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, .6, .7, .8, 0.9, 0.95, 1]) 
nGen = 1 
for p0 in p0_list:
    pops = [population(nInd, p0) for i in range(1000)] 
    [pop.evolve(nGen) for pop in pops]
    sample = [pop.get_frequency_trajectory()[-1] for pop in pops]
    variances.append(np.var(sample))
    histograms.append(plt.hist(sample, 100, alpha=0.5, range=(0,1)))
plt.xlabel("Population frequency")
plt.ylabel("Number of populations")
```

![png](image11.PNG)

Find the relationship between initial frequency and variance. You can deduce it from the math exercise above, look it up on wikipedia, but you can also just try to guess it from the simulations.

Tips for guessing: 

First, make the plot of variance vs frequency below

Then consider how much variance there is for p0=0 and p0=1.  

Can you come up with a simple function that has these values? Hint: it's a very simple function! 

```diff
@@ Answer : The relationship between initial frequency and variance is : Variance(p0) = p0 * (1-p0) / N. Variance for p0 = 0 and po = 1 is zero from the above formula. Also it is apparent in the above graph. @@ 
```

Can you come up with a simple function that has these values? Hint: it's a very simple function!
```python
### 22

plt.plot(np.array(p0_list), variances, 'o', label="simulations")
plt.plot(np.array(p0_list), [x*(1-x)/nInd for x in p0_list], '-', label="theory")  # Your theory.  
plt.ylabel("Variance")
plt.xlabel(r"initial frequency p_0")
plt.legend()
```

![png](image12.PNG)


Can you explain why this function is symmetrical around $p_0=0.5? 

```diff
@@ Answer : Algebraically, the function is symmetrical around $p_0=0.5 if f(p0) = f(1-p0). We saw that f(p0) = p0(1-p0)/N. Let p0 be replaced by (1-p0): f(1-p0) = (1-p0)(1-1+p0)/N = (1-p0)p0/N = f(p0) @@
```

## Mutation
New mutations enter the population in a single individual, and therefore begin their journey at frequency $\frac{1}{N}$. Numerically estimate the probability that such a new mutation will eventually fix (i.e., the probability that the mutation reaches frequency 1) in the population, if no subsequent mutations occur. 




```python
### 23

nInd = 10
nGen = 100
nRuns = 2000
# Enter the initial allele frequency for new mutations
p0 = 1/nInd
pops = [population(nInd,p0) for i in range(nRuns)] 
[pop.evolve(nGen) for pop in pops]; 
```

We can plot the number of populations at each frequency, as we did above.


```python
### 24
nBins = nInd + 1  # We want to have bins for 0,1,2,..., N copies of the allele. 
proportion_fixed = []  # fixation rate
for i in range(nGen+1):
    x,y = frequencyAtGen(i, pops, nBins);     
    if i==0:
        plt.plot(x, y, color=plt.cm.autumn(i*1./nGen), label="first generation")  # cm.autumn(i*1./nGen) returns the 
                                                                            #color with which to plot the current line
    elif i==nGen:
        plt.plot(x, y, color=plt.cm.autumn(i*1./nGen), label="generation %d"% (nGen,) )
    else:
        plt.plot(x, y, color=plt.cm.autumn(i*1./nGen))
    
    # We'll consider a population "fixed" if it is in the highest-frequency bin. It's
    # an approximation, but not a bad one if the number of bins is comparable to the 
    # population size.
    proportion_fixed.append((i, y[-1]*1./nRuns))
    
plt.legend()    
plt.xlabel("Population frequency")
plt.ylabel("Number of simulations")
```


![png](image13.PNG)


Here you should find that most mutations fix at zero frequency--only a small proportion survives. 

*What is the probability that a new mutation fixes in the population?--I'd like you to solve this problem both mathematically and numerically.?*.  

The mathematical part requires almost no calculation or mathematical knowledge, once you think about it in the right way. You can include your mathematical solution in the box below.  

```diff
@@ Answer : Let consider a haploid population of N individuals. If a mutation is introduced in a single individual its allele frequency is 1/N. If we consider a single individual’s allele over very long period of time there can be only two possibilities: it either fixes in the population or becomes extinct. Since there is no selection the chance for any individual to produce a surviving lineage is equal. So the probability that the mutation fixes in the population is its allele frequency. That is 1/N @@
```

For the computational part, note that we already computed the proportion of fixed alleles vs time in the "proportion_fixed" variable. So if we simulate long enough, we'll find the proportion of mutations that eventually fix.
Make sure that the numerical value agrees with the mathematical expectation!

```python
### 25
proportion_fixed = np.array(proportion_fixed)
fixation_probability = [p0]*(nGen+1)
plt.plot(proportion_fixed[:,0], np.array(fixation_probability), '--', label='theory 1/N')
plt.plot(proportion_fixed[:,0], proportion_fixed[:,1], '-', label="simulation")
plt.xlabel("Generation")
plt.ylabel("Fixation probability")
plt.legend()
```

![png](image14.PNG)

```diff
@@ As we can see the fraction of population where mutation fixed converges to the predicted value.@@
```


# Summary

Some important things that we've seen in this notebook:
* The Wright-Fisher model. Despite its simplicity, it is the basic building block of a large fraction of population genetics.
* In finite populations, sampling fluctuations are an important driver of allele frequency change.
* These sampling fluctuations cause larger frequency changes in smaller populations.
* These fluctuations mean that alleles eventually fix one way or another -- We need new mutations to maintain diversity within a population.
* For neutral alleles, the probability of new mutations fixing in the population is inversely proportional to the population size

# Something to think about. 

We'll get to selection, recombination, and linkage in the next exercises. In the meantime, you can think about the following:

* How much time will it take for a typical new mutation to reach fixation for different population sizes?
* If you add a constant influx of new mutations, how will the distribution of allele frequency look like at any given point in time?

Copyright: Simon Gravel. Do not share of distribute this file without the written consent of the author.
