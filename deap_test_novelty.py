import random
import utils

from deap import base
from deap import creator
from deap import tools
from sklearn.neighbors import NearestNeighbors
from itertools import repeat
import numpy as np

CONST_TOTAL_TEST_STATEMENTS = 3020.0
CONST_POPULATION_SIZE = 50

files, cases = utils.parse_and_build_test_case_data()

CONST_TOTAL_TEST_SUITE_COUNT = len(cases)

global_novelty_archive_list = dict()
global_best_individuals = [[[0] * 200, 0]]

# each gen
# run_custom_test_suite_and_calculate_test_coverage(cases, random_activated_cases_list)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, CONST_TOTAL_TEST_SUITE_COUNT)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual, pop):
    utils.run_custom_test_suite_and_calculate_test_coverage(
            cases, individual)
    coverage_value = CONST_TOTAL_TEST_STATEMENTS-utils.parse_coverage_report()[1]

    flag = 1
    for x in global_best_individuals:
        if coverage_value < x[1]:
            flag = 0
            break
    
    if flag:
        global_best_individuals.append([individual, coverage_value])

    # if coverage_value > novelty_best_individual[1]:
    #     global_best_individual = [individual, coverage_value]

    if len(global_novelty_archive_list) < CONST_POPULATION_SIZE:
        global_novelty_archive_list[tuple(individual)] = (1, coverage_value)
        return 1, 
    else:
        # Calculate distance against every individual in the current pop
        # Calculate distance against every individual in the archive
        # choose knn average dist.

        # print("pop")
        # print(pop)

        population_and_novelty_list = [list(k) for k, v in global_novelty_archive_list.items()] + pop

        # print("population_and_novelty_list")
        # print(population_and_novelty_list)

        knn_calculator = NearestNeighbors(n_neighbors = 3).fit(
            population_and_novelty_list)
        
        distances, indices = knn_calculator.kneighbors([individual])

        individual_novelty_metric = np.mean(distances)

        # print(individual_novelty_metric)

        if individual_novelty_metric > 7:
            global_novelty_archive_list[tuple(individual)] = (
                individual_novelty_metric, coverage_value)

        return individual_novelty_metric,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def main():
    # random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=CONST_POPULATION_SIZE)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop, repeat(pop)))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < CONST_TOTAL_TEST_STATEMENTS and g < 50:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind, repeat(pop))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_individual = [[0], 0]
        for x in global_best_individuals:
            if x[1] > best_individual[1]:
                 best_individual = x
        activated_test_cases = sum(best_individual[0])
        coverage = best_individual[1]/3020.0 * 100
        print("  best agent's no of test cases: {}".format(activated_test_cases))
        print("  total number of test cases minimized: {} (from {})".format(CONST_TOTAL_TEST_SUITE_COUNT-activated_test_cases, CONST_TOTAL_TEST_SUITE_COUNT))
        print("  best agent's coverage %: {}".format(coverage))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
