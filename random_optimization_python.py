
# coding: utf-8

# # random optimization algos implementations for cs 7641
# 
# this notebook provides simple implementations of random restart hill climbing, simulated annealing, genetic algorithms, and mimic, on **bitstrings only**.
# 
# Usage:
# 
#     rhc(bitstring_length, fitness_function)
#     sa(bitstring_length, fitness_function)
#     ga(bitstring_length, fitness_function)
#     mimic(bitstring_length, fitness_function)
#     
# Certain hyperparameters for each algorithm are also available. 
# 
# Requires tensorflow for MIMIC, but the network is simple enough that it should run on CPU just fine, and still way faster than the very slow alternative (https://github.com/mjs2600/mimicry). Please note that the method of modeling the probability distribution is very different from the one presented in the lectures / mimic paper (IMO, this approach is much better, more expandable, and faster).
# 
# All algorithms use a "patience" parameter to determine when to stop. Since they don't know if the current maximum is the global maximum, and give up if they don't find a better result within patience tries. 

# In[28]:

import numpy as np
import random, numpy
from deap import base, creator, tools, algorithms
import tensorflow as tf, numpy as np


# In[29]:

def rand_neighbor(bitstring):
    idx = random.randint(0, len(bitstring)-1)
    bitstring = bitstring.copy()
    bitstring[idx] = 1 - bitstring[idx]
    return bitstring


# # implement hc

# In[31]:

def hill_climb(bitstring, evalfn, patience=50):
    """Hill climbing applied to `bitstring` using fitness function `evalfn`. 
    Gives up trying to climb after `patience` random neighbors fail to produce
    a better maximum."""
    no_improvement = 0
    current_fitness = evalfn(bitstring)
    while True:
        candidate = rand_neighbor(bitstring)
        fitness = evalfn(candidate)
        if evalfn(candidate) > current_fitness:
            bitstring = candidate
            current_fitness = fitness
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            break
    return bitstring, current_fitness

def rhc(stringlen, evalfn, patience=20):
    """Random restart HC applied to bitstrings of length `stringlen` with 
    fitness function `evalfn`. Restarts until `patience` consecutive restarts
    fail to produce a better maximum."""
    no_improvement = 0
    best_bitstring = np.random.randint(0, 2, [stringlen])
    best_fitness = evalfn(best_bitstring)
    while True:
        bitstring = np.random.randint(0, 2, [stringlen])
        candidate, fitness = hill_climb(bitstring, evalfn)
        if fitness > best_fitness:
            best_bitstring = candidate
            best_fitness = fitness
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            break
    return best_bitstring, best_fitness            


# In[33]:

rhc(20, sum)


# # implement sa

# In[34]:

def sa(stringlen, evalfn, init_temp=1., decay=0.99, patience=1000):
    no_improvement = 0
    best_bitstring = np.random.randint(0, 2, [stringlen])
    best_fitness = evalfn(best_bitstring)
    
    bitstring = best_bitstring.copy()
    current_fitness = best_fitness
    temp = init_temp
    while True:
        candidate = rand_neighbor(bitstring)
        fitness = evalfn(candidate)
        
        if fitness > current_fitness or random.random() < np.exp((fitness - current_fitness) / temp):
            bitstring = candidate
            current_fitness = fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_bitstring = bitstring
                no_improvement = -1            
        
        temp *= decay
        no_improvement += 1    
        if no_improvement >= patience:
            break
            
    return best_bitstring, best_fitness


# In[36]:

sa(20, sum)


# # implement ga

# In[37]:

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# In[38]:

def ga(stringlen, evalfn, pop=50, cxpb=0.5, mutpb=0.2, patience=10):
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=stringlen)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
       
    def evalReturningTuple(x):
        return (evalfn(x),)
    
    toolbox.register("evaluate", evalReturningTuple)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    def run_5_gens(pop, hof):
        if pop is None:
            pop = toolbox.population(n=50)
        if hof is None:
            hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=False)

        return pop, hof
    
    no_improvement = 0
    best_fitness = float('-inf')
    pop, hof = None, None
    while True:
        pop, hof = run_5_gens(pop, hof)
        fitness = hof[0].fitness.values[0]
        if fitness > best_fitness:
            best_fitness = fitness
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            break
    return hof[0], hof[0].fitness.values[0]


# In[40]:

ga(20, sum)


# # implement mimic

# In[41]:

def mimic(stringlen, evalfn, latentlen = 100, num_samples = 100, samples_to_keep = 50, patience=20):
    batchsize = tf.placeholder(tf.int32)
    r = tf.random_uniform([batchsize, latentlen])
    logits = tf.layers.dense(r, stringlen)
    labels = tf.placeholder(tf.float32, shape=[None, stringlen])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    generated_samples = tf.floor(tf.nn.sigmoid(logits) + tf.random_uniform([batchsize, stringlen]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    best_fitness = float('-inf')
    bitstring = None
    no_improvement = 0
    while True: 
        samples = sess.run(generated_samples, feed_dict={batchsize: num_samples})
        fitnesses = np.array(list(map(evalfn, samples)))
        sorted_fitnesses = np.argsort(fitnesses)
        kept_samples = samples[sorted_fitnesses][-samples_to_keep:]
        fitness = evalfn(kept_samples[-1])
        
        if fitness > best_fitness:
            bitstring = kept_samples[-1]
            best_fitness = fitness
            no_improvement = -1            
        
        no_improvement += 1    
        
        if no_improvement >= patience:
            break
            
        sess.run(train_step, feed_dict={batchsize: samples_to_keep, labels: kept_samples})
    
    sess.close()
    return bitstring, best_fitness


# In[43]:

# mimic(20, sum)


# In[ ]:



