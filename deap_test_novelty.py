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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, CONST_TOTAL_TEST_SUITE_COUNT)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

    if len(global_novelty_archive_list) < CONST_POPULATION_SIZE:
        global_novelty_archive_list[tuple(individual)] = (1, coverage_value)
        return 1, 
    else:
        population_and_novelty_list = [list(k) for k, v in global_novelty_archive_list.items()] + pop

        knn_calculator = NearestNeighbors(n_neighbors = 3).fit(
            population_and_novelty_list)
        
        distances, indices = knn_calculator.kneighbors([individual])

        individual_novelty_metric = np.mean(distances)

        if individual_novelty_metric > 7:
            global_novelty_archive_list[tuple(individual)] = (
                individual_novelty_metric, coverage_value)

        return individual_novelty_metric,

toolbox.register("evaluate", evalOneMax)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)

toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # random.seed(64)

    pop = toolbox.population(n=CONST_POPULATION_SIZE)

    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop, repeat(pop)))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    fits = [ind.fitness.values[0] for ind in pop]

    g = 0

    while max(fits) < CONST_TOTAL_TEST_STATEMENTS and g < 50:
        g = g + 1
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind, repeat(pop))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

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
