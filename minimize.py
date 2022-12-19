"""
University of Ottawa
CSI5137[B] AI-Enabled Software Verification and Testing
Prof. Shiva Nejati

--------------------------------------------------------------------------------

Course Project
A Novelty Search Algorithm Approach for Test Suite Minimization

--------------------------------------------------------------------------------

Vignesh Kumar Karuppasamy - 300274799
Durga Devi Sivakumar -  300281361
"""

import argparse
import csv
from itertools import repeat
import random
import re
import subprocess
import os

from deap import base, creator, tools
import numpy as np
from sklearn.neighbors import NearestNeighbors

def parse_and_build_test_case_data(test_suite_path):
    """
    This function builds a list of test cases present in a test suite.
    
    Args:
        test_suite_path (str): The path to the test suite directory.
        
    Returns:
        tuple: A sorted list of test files and test cases
    """
    build_test_case_cmd = "pytest --collect-only {}".format(test_suite_path)

    build_test_case_output = subprocess.run(
        build_test_case_cmd, shell=True, capture_output=True, text=True
    )

    # if command failed, then return with error
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

    # iterate through each line in the output
    for line in build_test_case_output.splitlines():
        test_suit_file_result = re.search(regex_test_suite_file, line)

        # match for test suite files
        if test_suit_file_result:
            test_suite_file = test_suit_file_result.group(1)
            test_suite_files.add(test_suite_file)
            current_test_suite_file = test_suite_file

        if current_test_suite_file:
            # match for test classes
            test_class_result = re.search(regex_test_class, line)

            if test_class_result:
                current_test_class = test_class_result.group(1)

            # match for test cases
            test_case_result = re.search(regex_test_case_params, line)

            if not test_case_result:
                test_case_result = re.search(regex_test_case, line)
            
            # match for nested test cases inside test classes
            test_class_test_case_result = re.search(
                regex_test_class_test_case_params, line)

            if not test_class_test_case_result:
                test_class_test_case_result = re.search(
                    regex_test_class_test_case, line)

            # build test string for pytest runs
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
    test_suite_path, test_cases_list=None, test_cases_activation_list=None):
    """
    This function runs a test suite with coverage measurement

    It can take a list of test cases to run, but by default it will run all
    the test cases present in the test suite.

    Args:
        test_suite_path (str): The path to the test suite to be run.
        test_cases_list (list of str, optional): path of test cases to be run
        test_cases_activation_list (list of bool, optional):
            whether the corresponding test case should be run or not
        
    Returns:
        bool: whether the run was successful or not
    """
    test_cases_to_run = ""

    test_project_path = os.path.dirname(test_suite_path)

    if test_cases_list and test_cases_activation_list:
        for test_case_activation in zip(
                test_cases_list, test_cases_activation_list):
            if test_case_activation[1]:
                test_cases_to_run += (
                    f" {test_project_path}/{test_case_activation[0]}")
    else:
        test_cases_to_run += " {}".format(test_suite_path)

    coverage_run_cmd = "coverage run --branch -m pytest{}".format(
        test_cases_to_run)

    coverage_run_cmd_output = subprocess.run(
        coverage_run_cmd, shell=True, capture_output=True, text=True
    )

    if coverage_run_cmd_output.returncode != 0:
        return False

def parse_coverage_report():
    """
    This function parses the coverage report from the last coverage run and
    returns the coverage stats

    Returns:
        tuple: Statement coverage total and missed. Branch coverage total and 
        missed.
    """
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
    """
    This function takes in a list of test case paths and randomly activates
    some of them based on a random probability.

    Returns:
        list: list of boolean values corresponding to the activated test cases
    """
    return [random.choice([0, 1]) for x in test_cases_list]


class TestSuiteMinimization:
    def __init__(
        self, test_suite_path, search_method, test_suite_cases,
        total_test_statements, population_size=100, 
        max_generation_count=10000,  cross_over_probability=0.5, 
        mutation_probability=0.2, mutation_attribute_probability=0.05, 
        selection_tournament_size=3, distance_metric='hamming', 
        novelty_archive=True):
        """
        This class holds variables and functions used for performing test case
        minimization using Genetic Algorithm, as well as for Novelty Search.
        It also contains helper/utility functions for performing these tasks.

        Args:
            test_suite_path (str): the path to the test suite to be minimized
            search_method: the evolutionary search method to be used
            test_suite_cases (list of str): path to the test cases
            total_test_statements (int): total number of statements covered
            population_size (int, optional): population size for running
                genetic algorithms (default is 100)
            max_generation_count (int, optional): maximum generations to run
                genetic algorithms for (default is 10000)
            cross_over_probability (float, optional): probability of crossover
                (default is 0.5)
            mutation_probability (float, optional): probability of mutation
                (default is 0.2)
            mutation_attribute_probability (float, optional): probability of 
                mutation for an attribute(default is 0.05)
            selection_tournament_size (int, optional): size of the tournament 
                selection used during each generation (default is 3)
            distance_metric (str, optional): distance metric to use for 
                calculating k-nearest neighbors (default is 'hamming')
            novelty_archive (bool, optional): boolean indicating whether to use 
                a novelty archive (default is True)
        """
        self.CONST_TEST_SUITE_PATH = test_suite_path
        self.CONST_SEARCH_METHOD = search_method
        self.CONST_TEST_SUITE_CASES = test_suite_cases
        self.CONST_TOTAL_TEST_STATEMENTS = total_test_statements
        self.CONST_POPULATION_SIZE = population_size
        self.CONST_MAX_GENERATION_COUNT = max_generation_count
        self.CONST_CROSS_OVER_PROBABILITY = cross_over_probability
        self.CONST_MUTATION_PROBABILITY = mutation_probability
        self.CONST_MUTATION_ATTRIBUTE_PROBABILITY = \
                mutation_attribute_probability
        self.CONST_SELECTION_TOURNAMENT_SIZE = selection_tournament_size
        self.KNN_DISTANCE_METRIC = distance_metric
        self.NOVELTY_ARCHIVE = novelty_archive

        # if novelty method selected, then store some global variables
        if self.CONST_SEARCH_METHOD == "novelty":
            # deactivate archive if option is selected
            if self.NOVELTY_ARCHIVE:
                self.global_novelty_archive_list = dict()
            self.global_best_individuals = [
                [[0] * len(self.CONST_TEST_SUITE_CASES), 0]]

    def print_minimization_params(self):
        """
        This function prints the paramaters that TestSuiteMinimization was
        created and run with.
        """
        print("-----------------------------------")
        print("Test Suite Minimization Initialized")
        print("-----------------------------------")
        print("Test Suite Path: {}".format(self.CONST_TEST_SUITE_PATH))
        print("Minimization Method: {}".format(self.CONST_SEARCH_METHOD))
        print("Test Suite Size: {} test cases".format(
                len(self.CONST_TEST_SUITE_CASES)))
        print("Test Suite Total Statements: {}".format(
                self.CONST_TOTAL_TEST_STATEMENTS))
        print("Population Size: {}".format(self.CONST_POPULATION_SIZE))
        print("Max Generations: {}".format(self.CONST_MAX_GENERATION_COUNT))
        print("Crossover Probability: {}".format(
                self.CONST_CROSS_OVER_PROBABILITY))
        print("Mutation Probability: {}".format(
                self.CONST_MUTATION_PROBABILITY))
        print("Attribute Mutation Probability: {}".format(
                self.CONST_MUTATION_ATTRIBUTE_PROBABILITY))
        print("Tournament Selection Size: {}".format(
                self.CONST_SELECTION_TOURNAMENT_SIZE))
        if self.CONST_SEARCH_METHOD == "novelty":
            print("Use Novelty Archive: {}".format(self.NOVELTY_ARCHIVE))
            print("kNN Distance Metric: {}".format(self.KNN_DISTANCE_METRIC))
        print("-----------------------------------")

    def genetic_fitness_function(self, individual):
        """
        This function represents the fitness function to be used for evaluating
        individuals of a population during the genetic algorithm.

        Args:
            individual (list of bools): vector array representing a test suite

        Returns:
            int: the fitness value of the individual
        """
        run_custom_test_suite_and_calculate_test_coverage(
                self.CONST_TEST_SUITE_PATH, self.CONST_TEST_SUITE_CASES, 
                individual)
        return self.CONST_TOTAL_TEST_STATEMENTS-parse_coverage_report()[1],

    def perform_genetic_algorithm(self):
        """
        This function performs the genetic algorithm implementation after
        setup of the TestSuiteMinimization object.
        """
        print("Starting test suite minimization with genetic algorithm")

        creator.create(
                "GeneticAlgoFitness", base.Fitness, weights=(1.0,))
        creator.create(
                "Individual", list, fitness = creator.GeneticAlgoFitness)

        genetic_toolbox = base.Toolbox()

        # an individual contains bools of 0 or 1
        genetic_toolbox.register("IndividualVector", random.randint, 0, 1)

        # an individual is created with a vector of such bools
        # the length is the number of test cases present in the suite
        genetic_toolbox.register(
            "individual", tools.initRepeat, creator.Individual, 
            genetic_toolbox.IndividualVector, len(self.CONST_TEST_SUITE_CASES))

        # a population is initialized with n number of these individuals
        genetic_toolbox.register(
            "population", tools.initRepeat, 
            list, genetic_toolbox.individual)

        # the genetic fitness function previously defined is used for 
        # evaluating each individual
        genetic_toolbox.register("evaluate", self.genetic_fitness_function)

        # the two point crossover method is used for crossover of individuals
        genetic_toolbox.register("mate", tools.cxTwoPoint)

        # mutation is performed by flipping the bools of the individual based
        # on a set probability
        genetic_toolbox.register(
            "mutate", tools.mutFlipBit,
            indpb = self.CONST_MUTATION_ATTRIBUTE_PROBABILITY)

        # selection is done through tournament size equal to n
        genetic_toolbox.register(
            "select", 
            tools.selTournament, 
            tournsize = self.CONST_SELECTION_TOURNAMENT_SIZE)

        current_population = genetic_toolbox.population(
            n = self.CONST_POPULATION_SIZE)

        # calculate fitness for initial population
        current_population_fitness = list(
                map(genetic_toolbox.evaluate, current_population))
        for individual, fitness in zip(
                    current_population, current_population_fitness):
            individual.fitness.values = fitness

        fitness_values = [
            individual.fitness.values[0] for individual in current_population]

        current_generation_count = 0
        best_individual_test_suite, best_individual_coverage_value = None, None

        # run implementation for each generation requried
        while (
                max(fitness_values) < self.CONST_TOTAL_TEST_STATEMENTS and 
                current_generation_count < self.CONST_MAX_GENERATION_COUNT):
            current_generation_count = current_generation_count + 1
            print("Generation #{}".format(current_generation_count))

            # clone the current population
            current_generation_offspring = genetic_toolbox.select(
                current_population, len(current_population))
            current_generation_offspring = list(
                map(genetic_toolbox.clone, current_generation_offspring))

            # perform crossover
            for offspring_1, offspring_2 in zip(
                        current_generation_offspring[::2], 
                        current_generation_offspring[1::2]):
                if random.random() < self.CONST_CROSS_OVER_PROBABILITY:
                    genetic_toolbox.mate(offspring_1, offspring_2)

                    del offspring_1.fitness.values
                    del offspring_2.fitness.values

            # perform mutation
            for offpsring in current_generation_offspring:
                if random.random() < self.CONST_MUTATION_PROBABILITY:
                    genetic_toolbox.mutate(offpsring)
                    del offpsring.fitness.values

            # recalculate fitness for mutated individuals
            invalid_fitness_individuals = [
                    individual for individual in current_generation_offspring \
                    if not individual.fitness.valid]
            current_population_fitness = map(
                    genetic_toolbox.evaluate, invalid_fitness_individuals)
            for individual, fitness in zip(
                        invalid_fitness_individuals, 
                        current_population_fitness):
                individual.fitness.values = fitness

            # set population for next iteration
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
            
            best_individual_test_suite, best_individual_coverage_value = (
                tools.selBest(current_population, 1)[0], 
                coverage)

            print("Average Fitness: {}".format(average_fitness))
            print("Best Individual Stats: ")
            print("No. of Test Cases: {}".format(activated_test_cases))
            print("No. of test cases minimized: {} (from {})".format(
                len(self.CONST_TEST_SUITE_CASES)-activated_test_cases, 
                len(self.CONST_TEST_SUITE_CASES)))
            print("Coverage value: {}%".format(coverage))

        print("Finished evluating test suite using genetic algorithm!")

        return best_individual_test_suite, best_individual_coverage_value

    def calculate_novelty_metric(self, individual, population):
        """
        This function represents the novelty metric to be used for evaluating
        the behavior of individuals of a population for novelty search.

        Args:
            individual (list of bools): vector array representing a test suite
            population (list of individuals): the total current population

        Returns:
            int: the novelty value of the individual
        """
        # calculate the coverage values for the current individual
        run_custom_test_suite_and_calculate_test_coverage(
                self.CONST_TEST_SUITE_PATH, self.CONST_TEST_SUITE_CASES, 
                individual)
        coverage_value = self.CONST_TOTAL_TEST_STATEMENTS - \
                parse_coverage_report()[1]

        flag = 1
        for x in self.global_best_individuals:
            if coverage_value < x[1]:
                flag = 0
                break

        # store the value if it is the best individual encountered so far
        if flag:
            self.global_best_individuals.append([individual, coverage_value])

        # novelty archive is selected, then store very novel individuals
        # in a global archive list
        if self.NOVELTY_ARCHIVE:
            # fill up the novelty archive with all individuals of the first
            # generation
            if len(self.global_novelty_archive_list) < \
                    self.CONST_POPULATION_SIZE:
                self.global_novelty_archive_list[
                        tuple(individual)] = (1, coverage_value)
                return 1,
            else:
                # combine archive and current population
                population_and_novelty_list = [
                    list(k) for k, _ in \
                        self.global_novelty_archive_list.items()] \
                    + population

                # perform k-nearest neighbor calculation
                knn_calculator = NearestNeighbors(
                        n_neighbors = 3,
                        metric = self.KNN_DISTANCE_METRIC).fit(
                            population_and_novelty_list)
                
                distances, _ = knn_calculator.kneighbors([individual])

                # find the average distance to k-neighbors
                individual_novelty_metric = np.mean(distances)

                # if novelty metric for current individual is above threshold,
                # add it to the global novel archive list
                if individual_novelty_metric > 7:
                    self.global_novelty_archive_list[tuple(individual)] = (
                        individual_novelty_metric, coverage_value)

                return individual_novelty_metric,
        else:
            # if novel archive is not selected, then perform k-nn and
            # return the average distance to the k-neighbors
            knn_calculator = NearestNeighbors(
                        n_neighbors = 3,
                        metric = self.KNN_DISTANCE_METRIC).fit(
                            population)
                
            distances, _ = knn_calculator.kneighbors([individual])

            individual_novelty_metric = np.mean(distances)

            return individual_novelty_metric,

    def perform_novelty_search(self):
        """
        This function performs the novelty search implementation after
        setup of the TestSuiteMinimization object.
        """
        print("Starting test suite minimization with novelty search")

        creator.create("NoveltyMetric", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.NoveltyMetric)

        novelty_toolbox = base.Toolbox()

        novelty_toolbox.register("attr_bool", random.randint, 0, 1)

        novelty_toolbox.register(
            "individual", tools.initRepeat, creator.Individual, 
            novelty_toolbox.attr_bool, len(self.CONST_TEST_SUITE_CASES))

        novelty_toolbox.register(
            "population", tools.initRepeat, list, novelty_toolbox.individual)

        novelty_toolbox.register("evaluate", self.calculate_novelty_metric)

        novelty_toolbox.register("mate", tools.cxTwoPoint)

        novelty_toolbox.register(
            "mutate", tools.mutFlipBit, 
            indpb = self.CONST_MUTATION_ATTRIBUTE_PROBABILITY)

        novelty_toolbox.register(
            "select", tools.selTournament, 
            tournsize=self.CONST_SELECTION_TOURNAMENT_SIZE)

        current_population = novelty_toolbox.population(
            n = self.CONST_POPULATION_SIZE)

        current_population_novelty = list(
            map(novelty_toolbox.evaluate, current_population, 
                repeat(current_population)))
        for individual, novelty in zip(
                    current_population, current_population_novelty):
            individual.fitness.values = novelty

        novelty_values = [
            individual.fitness.values[0] for individual in current_population]

        current_generation_count = 0

        best_individual_test_suite, best_individual_coverage_value = None, None

        while (
                max(novelty_values) < self.CONST_TOTAL_TEST_STATEMENTS and 
                current_generation_count < self.CONST_MAX_GENERATION_COUNT):
            current_generation_count = current_generation_count + 1
            print("Generation #{}".format(current_generation_count))

            current_generation_offspring = novelty_toolbox.select(
                    current_population, len(current_population))
            current_generation_offspring = list(
                map(novelty_toolbox.clone, current_generation_offspring))

            for offspring_1, offspring_2 in zip(
                        current_generation_offspring[::2], 
                        current_generation_offspring[1::2]):
                if random.random() < self.CONST_CROSS_OVER_PROBABILITY:
                    novelty_toolbox.mate(offspring_1, offspring_2)

                    del offspring_1.fitness.values
                    del offspring_2.fitness.values

            for offspring in current_generation_offspring:
                if random.random() < self.CONST_MUTATION_PROBABILITY:
                    novelty_toolbox.mutate(offspring)
                    del offspring.fitness.values

            invalid_novelty_individuals = [
                individual for individual in current_generation_offspring \
                    if not individual.fitness.valid]
            novelty_values = map(
                    novelty_toolbox.evaluate, invalid_novelty_individuals, 
                    repeat(current_population))
            for individual, novelty in zip(
                        invalid_novelty_individuals, novelty_values):
                individual.fitness.values = novelty

            current_population[:] = current_generation_offspring

            novelty_values = [
                individual.fitness.values[0] \
                for individual in current_population]

            population_size = len(current_population)
            average_fitness = sum(novelty_values) / population_size
            activated_test_cases = sum(
                    tools.selBest(current_population, 1)[0])
            coverage = tools.selBest(
                current_population, 1)[0].fitness.values[0] \
                / self.CONST_TOTAL_TEST_STATEMENTS * 100

            print("Average Novelty: {}".format(average_fitness))

            best_individual = [[0], 0]
            for x in self.global_best_individuals:
                if x[1] > best_individual[1]:
                    best_individual = x
            activated_test_cases = sum(best_individual[0])
            coverage = best_individual[1]/self.CONST_TOTAL_TEST_STATEMENTS * 100

            best_individual_test_suite, best_individual_coverage_value = (
                    best_individual[0], coverage)
            print("Best Individual Stats: ")
            print("No. of Test Cases: {}".format(activated_test_cases))
            print("No. of test cases minimized: {} (from {})".format(
                len(self.CONST_TEST_SUITE_CASES)-activated_test_cases, 
                len(self.CONST_TEST_SUITE_CASES)))
            print("Coverage value: {}%".format(coverage))

        print("Finished evaluating test suite using Novelty Search!")

        return best_individual_test_suite, best_individual_coverage_value

    def perform_test_suite_minimization(self):
        """
        This generic function performs specific test suite minimization method
        based on the given parameters stored in the class object.
        """
        self.print_minimization_params()
        if self.CONST_SEARCH_METHOD == "genetic":
            return self.perform_genetic_algorithm()
        elif self.CONST_SEARCH_METHOD == "novelty":
            return self.perform_novelty_search()
        
        return False


if __name__ == "__main__":
    # parse the command line arguments for the program
    parser = argparse.ArgumentParser(
            description="Parse arguments for minimize.py")
    parser.add_argument('--path', metavar="test_suite_path", 
            type=str,
            required = True, 
            help="path to directory containing the test suite to be minimized")
    parser.add_argument('--algorithm', metavar="search_algorithm", 
            type=str,
            default = "genetic",
            required = False,
            help="selecting the algorithm for running test suite minimization")
    parser.add_argument('--pop_size', metavar="population_size", 
            type=int,
            required = False,
            default=100,
            help="the population size used in the genetic algorithms")
    parser.add_argument('--max_gen', metavar="max_generation_count", 
            type=int,
            required = False,
            default=10000,
            help="the maximum generation for iterations in the algorithms")
    parser.add_argument('--crossover', metavar="cross_over_probability", 
            type=float,
            required = False,
            default = 0.5,
            help="the cross over probability used in the genetic algorithms")
    parser.add_argument('--mutation', metavar="mutation_probability", 
            type=float,
            required = False,
            default = 0.2,
            help="the mutation probability used in the genetic algorithms")
    parser.add_argument('--attribute_mutation', 
            metavar="mutation_attribute_probability", 
            type=float,
            required = False,
            default = 0.05,
            help="mutation probability for the attribute of each individual")
    parser.add_argument('--tourn_size', metavar="selection_tournament_size", 
            type=int,
            required = False,
            default = 3,
            help="the tournament size criteria for performing selection")
    parser.add_argument('--distance_metric', metavar="distance_metric", 
            type=str,
            required = False,
            default = "hamming",
            help="the distance metric to use for calculating kNN in N.S.")
    parser.add_argument('--omit_novelty_archive',
            action="store_true",
            required = False,
            default = False,
            help="whether to use a novelty archive during Novelty Search")

    args = parser.parse_args()
    test_suite_path = args.path
    search_method = args.algorithm
    population_size = args.pop_size
    max_generation_count = args.max_gen
    cross_over_probability = args.crossover
    mutation_probability = args.mutation
    mutation_attribute_probability = args.attribute_mutation
    tournament_size = args.tourn_size
    distance_metric = args.distance_metric
    novelty_archive = not args.omit_novelty_archive

    # exit the program with error if test suite path is not present
    if not os.path.exists(test_suite_path):
        print("Could not find the specificed directory at the given path!")
        exit(1)

    # build list of cases present in the test suite
    test_suite_files, test_suite_cases_list = parse_and_build_test_case_data(
        test_suite_path=test_suite_path)

    # calculate initial coverage using total test suite
    run_custom_test_suite_and_calculate_test_coverage(
        test_suite_path=test_suite_path)

    total_statements, missed_statements, \
        branches, branches_missed = parse_coverage_report()
    
    full_test_suite_coverage = (
        (total_statements - missed_statements) / total_statements) * 100
    
    # define the test suite minimization object with the user's given values
    tsm = TestSuiteMinimization(
        test_suite_path=test_suite_path,
        search_method=search_method,
        test_suite_cases=test_suite_cases_list,
        total_test_statements=total_statements,
        population_size=population_size,
        max_generation_count=max_generation_count,
        cross_over_probability=cross_over_probability,
        mutation_probability=mutation_probability,
        mutation_attribute_probability=mutation_attribute_probability,
        selection_tournament_size=tournament_size,
        distance_metric=distance_metric,
        novelty_archive=novelty_archive)
    
    # perform the actual minimization
    best_test_suite, best_coverage_value = tsm.perform_test_suite_minimization()
    csv_file_path = 'minimized_test_suite.csv'

    # store the minimzed test suite to a .csv file in the same dir
    with open(csv_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Test Case", "Included"])

        for test_case, included in zip(test_suite_cases_list, best_test_suite):
            included = "YES" if included == 1 else "NO"
            csv_writer.writerow([test_case, included])

    print("-----------------------------------")
    print("Results")
    print("-----------------------------------")
    print("Minimized test suite contains: {} test cases".format(
        sum(best_test_suite)))
    print("Total test suite was reduced by: {} test cases (from {})".format(
        len(test_suite_cases_list) - sum(best_test_suite), 
        len(test_suite_cases_list)))
    print("Coverage reduction due to minimization: {:.2f}%".format(
        full_test_suite_coverage - best_coverage_value))
    print("Minimized test suite details saved to {}!".format(csv_file_path))
    print("-----------------------------------")
