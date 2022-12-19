import argparse
from itertools import repeat
import random
import re
import subprocess
import os

from deap import base, creator, tools
import numpy as np
from sklearn.neighbors import NearestNeighbors

def parse_and_build_test_case_data(test_suite_path):
    build_test_case_cmd = "pytest --collect-only {}".format(test_suite_path)

    build_test_case_output = subprocess.run(
        build_test_case_cmd, shell=True, capture_output=True, text=True
    )

    if build_test_case_output.returncode != 0:
        return False

    build_test_case_output = build_test_case_output.stdout

    test_suite_files = set()
    test_cases_list = set()

    # regexes
    regex_test_suite_file = "^<Module.* (.*.py)>$"
    regex_test_case = r"^  <Function (.*).*>$"
    regex_test_case_params = r"^  <Function (.*)\[.*>$"
    regex_test_class = "^  <Class (.*)>$"
    regex_test_class_test_case = r"^    <Function (.*).*>$"
    regex_test_class_test_case_params = r"^    <Function (.*)\[.*>$"

    current_test_suite_file = None
    current_test_class = None

    for line in build_test_case_output.splitlines():
        test_suit_file_result = re.search(regex_test_suite_file, line)

        if test_suit_file_result:
            test_suite_file = test_suit_file_result.group(1)
            test_suite_files.add(test_suite_file)
            current_test_suite_file = test_suite_file

        if current_test_suite_file:
            test_class_result = re.search(regex_test_class, line)

            if test_class_result:
                current_test_class = test_class_result.group(1)

            test_case_result = re.search(regex_test_case_params, line)

            if not test_case_result:
                test_case_result = re.search(regex_test_case, line)

            test_class_test_case_result = re.search(
                regex_test_class_test_case_params, line)

            if not test_class_test_case_result:
                test_class_test_case_result = re.search(
                    regex_test_class_test_case, line)

            if test_class_test_case_result:
                test_cases_list.add(
                    "{}::{}::{}".format(
                        current_test_suite_file,
                        current_test_class,
                        test_class_test_case_result.group(1),
                    )
                )

            if test_case_result:
                test_cases_list.add(
                    "{}::{}".format(
                        current_test_suite_file, test_case_result.group(1))
                )
                current_test_class = None

    return sorted(test_suite_files), sorted(test_cases_list)

def run_custom_test_suite_and_calculate_test_coverage(
    test_suite_path, test_cases_list=None, test_cases_activation_list=None
):
    test_cases_to_run = ""

    test_project_path = os.path.dirname(test_suite_path)

    if test_cases_list and test_cases_activation_list:
        for test_case_activation in zip(
                test_cases_list, test_cases_activation_list):
            if test_case_activation[1]:
                test_cases_to_run += (
                    f" {test_project_path}/{test_case_activation[0]}")

    coverage_run_cmd = "coverage run --branch -m pytest{}".format(
        test_cases_to_run)

    # print(coverage_run_cmd)

    coverage_run_cmd_output = subprocess.run(
        coverage_run_cmd, shell=True, capture_output=True, text=True
    )

    if coverage_run_cmd_output.returncode != 0:
        return False

def parse_coverage_report():
    coverage_report_cmd = "coverage report"

    coverage_report_cmd = subprocess.run(
        coverage_report_cmd, shell=True, capture_output=True, text=True
    )

    if coverage_report_cmd.returncode != 0:
        return False

    coverage_report_cmd = coverage_report_cmd.stdout

    coverage_stats_regex = r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+.*$"

    coverage_stats_result = re.search(
        coverage_stats_regex, coverage_report_cmd.splitlines()[-1]
    )

    if coverage_stats_result:
        statement_coverage_total = coverage_stats_result.group(1)
        statement_coverage_missed = coverage_stats_result.group(2)
        branch_coverage = coverage_stats_result.group(3)
        branch_coverage_missed = coverage_stats_result.group(4)

        return (
            int(statement_coverage_total),
            int(statement_coverage_missed),
            int(branch_coverage),
            int(branch_coverage_missed),
        )

    return False

def activate_random_test_suite(test_cases_list):
    return [random.choice([0, 1]) for x in test_cases_list]


class TestSuiteMinimization:
    def __init__(
            self, test_suite_path, test_suite_cases,
            total_test_statements, population_size=100, 
            max_generation_count=10000,  cross_over_probability=0.5, 
            mutation_probability=0.2, mutation_attribute_probability=0.05, 
            selection_tournament_size=3):
        self.CONST_TEST_SUITE_PATH = test_suite_path
        self.CONST_TEST_SUITE_CASES = test_suite_cases
        self.CONST_TOTAL_TEST_STATEMENTS = total_test_statements
        self.CONST_POPULATION_SIZE = population_size
        self.CONST_MAX_GENERATION_COUNT = max_generation_count
        self.CONST_CROSS_OVER_PROBABILITY = cross_over_probability
        self.CONST_MUTATION_PROBABILITY = mutation_probability
        self.CONST_MUTATION_ATTRIBUTE_PROBABILITY = \
                mutation_attribute_probability
        self.CONST_SELECTION_TOURNAMENT_SIZE = selection_tournament_size

    def genetic_fitness_algorithm(self, individual):
        run_custom_test_suite_and_calculate_test_coverage(
                self.CONST_TEST_SUITE_PATH, self.CONST_TEST_SUITE_CASES, 
                individual)
        return self.CONST_TOTAL_TEST_STATEMENTS-parse_coverage_report()[1],

    def perform_genetic_algorithm(self):
        print("Starting test suite minimization with genetic algorithm")

        creator.create(
                "GeneticAlgoFitness", base.Fitness, weights=(1.0,))
        creator.create(
                "Individual", list, fitness = creator.GeneticAlgoFitness)

        genetic_toolbox = base.Toolbox()

        genetic_toolbox.register("IndividualVector", random.randint, 0, 1)

        genetic_toolbox.register(
            "individual", tools.initRepeat, creator.Individual, 
            genetic_toolbox.IndividualVector, len(self.CONST_TEST_SUITE_CASES))

        genetic_toolbox.register(
            "population", tools.initRepeat, 
            list, genetic_toolbox.individual)

        genetic_toolbox.register("evaluate", self.genetic_fitness_algorithm)

        genetic_toolbox.register("mate", tools.cxTwoPoint)

        genetic_toolbox.register(
            "mutate", tools.mutFlipBit,
            indpb = self.CONST_MUTATION_ATTRIBUTE_PROBABILITY)

        genetic_toolbox.register(
            "select", 
            tools.selTournament, 
            tournsize = self.CONST_SELECTION_TOURNAMENT_SIZE)

        current_population = genetic_toolbox.population(
            n = self.CONST_POPULATION_SIZE)

        current_population_fitness = list(
                map(genetic_toolbox.evaluate, current_population))
        for individual, fitness in zip(
                    current_population, current_population_fitness):
            individual.fitness.values = fitness

        fitness_values = [
            individual.fitness.values[0] for individual in current_population]

        current_generation_count = 0

        while (
            max(fitness_values) < self.CONST_TOTAL_TEST_STATEMENTS and 
                current_generation_count < self.CONST_MAX_GENERATION_COUNT):
            current_generation_count = current_generation_count + 1
            print("Generation #{}".format(current_generation_count))

            current_generation_offspring = genetic_toolbox.select(
                current_population, len(current_population))
            current_generation_offspring = list(
                map(genetic_toolbox.clone, current_generation_offspring))

            for offspring_1, offspring_2 in zip(
                        current_generation_offspring[::2], 
                        current_generation_offspring[1::2]):
                if random.random() < self.CONST_CROSS_OVER_PROBABILITY:
                    genetic_toolbox.mate(offspring_1, offspring_2)

                    del offspring_1.fitness.values
                    del offspring_2.fitness.values

            for offpsring in current_generation_offspring:
                if random.random() < self.CONST_MUTATION_PROBABILITY:
                    genetic_toolbox.mutate(offpsring)
                    del offpsring.fitness.values

            invalid_fitness_individuals = [
                    individual for individual in current_generation_offspring \
                    if not individual.fitness.valid]
            current_population_fitness = map(
                    genetic_toolbox.evaluate, invalid_fitness_individuals)
            for individual, fitness in zip(
                        invalid_fitness_individuals, 
                        current_population_fitness):
                individual.fitness.values = fitness

            current_population[:] = current_generation_offspring

            fitnesses = [
                individual.fitness.values[0] \
                for individual in current_population]

            population_size = len(current_population)
            average_fitness = sum(fitnesses) / population_size
            activated_test_cases = sum(
                    tools.selBest(current_population, 1)[0])
            coverage = tools.selBest(
                current_population, 1)[0].fitness.values[0] \
                / self.CONST_TOTAL_TEST_STATEMENTS * 100

            print("Average Fitness: {}".format(average_fitness))
            print("Best Individual Stats: ")
            print("No. of Test Cases: {}".format(activated_test_cases))
            print("No. of test cases minimized: {} (from {})".format(
                len(self.CONST_TEST_SUITE_CASES)-activated_test_cases, 
                len(self.CONST_TEST_SUITE_CASES)))
            print("Coverage value: {}%".format(coverage))

        print("Finished evluating test suite using genetic algorithm!")

# def perform_novelty_search(test_suite_path):
#     global_novelty_archive_list = dict()
#     global_best_individuals = [[[0] * 200, 0]]

#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)

#     toolbox = base.Toolbox()

#     toolbox.register("attr_bool", random.randint, 0, 1)

#     toolbox.register("individual", tools.initRepeat, creator.Individual, 
#         toolbox.attr_bool, CONST_TOTAL_TEST_SUITE_COUNT)

#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#     def evalOneMax(individual, pop):
#         run_custom_test_suite_and_calculate_test_coverage(
#                 test_suite_path, cases, individual)
#         coverage_value = CONST_TOTAL_TEST_STATEMENTS-parse_coverage_report()[1]

#         flag = 1
#         for x in global_best_individuals:
#             if coverage_value < x[1]:
#                 flag = 0
#                 break
        
#         if flag:
#             global_best_individuals.append([individual, coverage_value])

#         if len(global_novelty_archive_list) < CONST_POPULATION_SIZE:
#             global_novelty_archive_list[tuple(individual)] = (1, coverage_value)
#             return 1, 
#         else:
#             population_and_novelty_list = [list(k) for k, v in global_novelty_archive_list.items()] + pop

#             knn_calculator = NearestNeighbors(n_neighbors = 3).fit(
#                 population_and_novelty_list)
            
#             distances, indices = knn_calculator.kneighbors([individual])

#             individual_novelty_metric = np.mean(distances)

#             if individual_novelty_metric > 7:
#                 global_novelty_archive_list[tuple(individual)] = (
#                     individual_novelty_metric, coverage_value)

#             return individual_novelty_metric,

#     toolbox.register("evaluate", evalOneMax)

#     toolbox.register("mate", tools.cxTwoPoint)

#     toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)

#     toolbox.register("select", tools.selTournament, tournsize=3)

#     pop = toolbox.population(n=CONST_POPULATION_SIZE)

#     CXPB, MUTPB = 0.5, 0.2

#     print("Start of evolution")

#     fitnesses = list(map(toolbox.evaluate, pop, repeat(pop)))
#     for ind, fit in zip(pop, fitnesses):
#         ind.fitness.values = fit

#     print("  Evaluated %i individuals" % len(pop))

#     fits = [ind.fitness.values[0] for ind in pop]

#     g = 0

#     while max(fits) < CONST_TOTAL_TEST_STATEMENTS and g < 50:
#         g = g + 1
#         print("-- Generation %i --" % g)

#         offspring = toolbox.select(pop, len(pop))
#         offspring = list(map(toolbox.clone, offspring))

#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < CXPB:
#                 toolbox.mate(child1, child2)

#                 del child1.fitness.values
#                 del child2.fitness.values

#         for mutant in offspring:
#             if random.random() < MUTPB:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values

#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = map(toolbox.evaluate, invalid_ind, repeat(pop))
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit

#         print("  Evaluated %i individuals" % len(invalid_ind))

#         pop[:] = offspring

#         fits = [ind.fitness.values[0] for ind in pop]

#         length = len(pop)
#         mean = sum(fits) / length
#         sum2 = sum(x*x for x in fits)
#         std = abs(sum2 / length - mean**2)**0.5

#         print("  Min %s" % min(fits))
#         print("  Max %s" % max(fits))
#         print("  Avg %s" % mean)
#         print("  Std %s" % std)
#         best_individual = [[0], 0]
#         for x in global_best_individuals:
#             if x[1] > best_individual[1]:
#                  best_individual = x
#         activated_test_cases = sum(best_individual[0])
#         coverage = best_individual[1]/3020.0 * 100
#         print("  best agent's no of test cases: {}".format(activated_test_cases))
#         print("  total number of test cases minimized: {} (from {})".format(CONST_TOTAL_TEST_SUITE_COUNT-activated_test_cases, CONST_TOTAL_TEST_SUITE_COUNT))
#         print("  best agent's coverage %: {}".format(coverage))

#     print("-- End of (successful) evolution --")

#     best_ind = tools.selBest(pop, 1)[0]
#     print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    # parse the command line arguments for the program
    parser = argparse.ArgumentParser(
            description="Parse arguments for minimize.py")
    parser.add_argument('--path', metavar="test_suite_path", 
            type=str,
            required = True, 
            help="Path to directory containing test suite")
    parser.add_argument('--algorithm', metavar="search_algorithm", 
            type=str,
            default = "genetic",
            required = False,
            help="selecting the algorithm for running test suite minimization")
    # parser.add_argument('--cool', metavar="cooling_factor", 
    #         type=float,
    #         default = 0.9999,
    #         required = False,
    #         help="cooling factor for performing SA algorithm")

    args = parser.parse_args()
    test_suite_path = args.path
    search_method = args.algorithm
    # temp = args.temp
    # cooling_factor = args.cool

    if not os.path.exists(test_suite_path):
        print("Could not find the specificed directory at the given path!")
        exit()

    # files, cases = parse_and_build_test_case_data(test_suite_path)
    # print(len(cases))
    # for file in files:
    #     print(file)

    # for case in cases:
    #     print(case)
    # random_activated_cases_list = activate_random_test_suite(cases)
    # print(random_activated_cases_list)
    # print(sum(random_activated_cases_list))
    # run_custom_test_suite_and_calculate_test_coverage(test_suite_path, cases, random_activated_cases_list)
    # report_values = parse_coverage_report()
    # print(report_values)
    # print(1.0-(report_values[1]/report_values[0]))

    files, cases = parse_and_build_test_case_data(test_suite_path)

    # CONST_TOTAL_TEST_SUITE_COUNT = len(cases)

    tsm = TestSuiteMinimization(
        test_suite_path=test_suite_path, test_suite_cases=cases,
        total_test_statements=3020.0)   
            
    if search_method == "genetic":
        tsm.perform_genetic_algorithm()
    
    # if search_method == "novelty":
    #     tsm.(test_suite_path = test_suite_path)
