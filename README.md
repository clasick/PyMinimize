# PyMinimize - A Python Test Suite Minimization Framework

Test suite minimization is a widely studied topic in the emerging artificial intelligence (AI) field of Search Based Software Testing (SBST). The aim of test suite minimization is to produce a sub-set of test cases from a test suite which reduces suite execution time, while having a negligible decrease in code coverage. 

Current state-of-the-art solutions exist only in Java-based test suites. The PyMinimize framework aims to bridge this gap by providing the only known Python framework for performing test suite minimization. It is also the only framework to use the Novelty Search (NS) algorithm for performing the minimization procedure.

Currently, the PyMinimize framework supports the following heuristic search algorithms:
1. Genetic Algorithm (GA)
2. Novelty Search (NS) Algorithm

The implementation of various parts of the heuristic search methods are implemented using the [Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap) (DEAP) project. The PyMinimize framework is quite generic, and is compatible with any Python project which uses the `pytest` module for running unit tests-- however, the framework has been only tested with the `Flask` Python project thus far.

## Dependencies
1. Deap
2. Scikit-learn
3. Numpy
4. Pytest
5. Coverage.py module

## General Info
1. The main file in the PyMinimize framework is the `minimize.py` file, which is a command line tool.
2. Run the command `minimize.py --help` command to see the full list of parameters that can be modified for running the evolutionary search algorithms. Primarily, you can modify the algorithm used through the `--algorithm` switch. The values this switch takes are `genetic` and `novelty`.
3. The test suites need to comply with the Pytest test bench framework and the Coverage.py tool. They need to be placed inside the `/test-suites` folder. A sample project `flask` is provided in the framework.
4. Dependencies of both minimize.py and the test suite that need to be run need to be installed in the current environment (for example, using anaconda or virtualenv). The pip command is used for installing the dependencies. **NOTE**: Failing to properly install these dependencies will cause unexpected results while running PyMinimize
5. As the search is randomly seeded, it will produce different evaluation on each run. The average time for the run's completion depends on the performance capabilities of the CPU it is run on.
6. Tested on Python 3.10.8 with an Anaconda environment running on macOS X 13.0.1

## Command line options
```
usage: minimize.py [-h] --path test_suite_path [--algorithm search_algorithm] 
[--pop_size population_size] [--max_gen max_generation_count] 
[--crossover cross_over_probability] [--mutation mutation_probability] 
[--attribute_mutation mutation_attribute_probability] 
[--tourn_size selection_tournament_size] [--distance_metric distance_metric] 
[--omit_novelty_archive]

Parse arguments for minimize.py

options:
  -h, --help            show this help message and exit
  --path test_suite_path
                        path to directory containing the test suite to be minimized
  --algorithm search_algorithm
                        selecting the algorithm for running test suite minimization
  --pop_size population_size
                        the population size used in the genetic algorithms
  --max_gen max_generation_count
                        the maximum generation for iterations in the algorithms
  --crossover cross_over_probability
                        the cross over probability used in the genetic algorithms
  --mutation mutation_probability
                        the mutation probability used in the genetic algorithms
  --attribute_mutation mutation_attribute_probability
                        mutation probability for the attribute of each individual
  --tourn_size selection_tournament_size
                        the tournament size criteria for performing selection
  --distance_metric distance_metric
                        the distance metric to use for calculating kNN in N.S.
  --omit_novelty_archive
                        whether to use a novelty archive during Novelty Search
```

## Instructions to Run
1. Navigate to the PyMinimize directory using the command line.
2. Ensure the presence of the file `minimize.py` in the directory, and the presence of the Flask folder in the `test-suites` directory.
3. Install dependencies for the Flask test suite by running the following commands:
a. `pip install -r test-suites/flask/requirements/dev.txt` 
b. `pip install -e test-suites/flask/`
c. Ensure the dependencies are installed successfully
4. Install the dependencies for minimize.py by running the command:
a. `pip install deap scikit-learn numpy`
b. Ensure the dependencies are installed successfully 
5. Run the `minimize.py` file with appropriate parameters.
a. For example, to run genetic algorithm for test suite minimization on the Flask project, use the following command:
i. `python minimize.py --path test-suites/flask/tests --pop_size 50 --max_gen 100 --crossover 0.5 --mutation 0.2 --attribute_mutation 0.10 --tourn_size 3 --algorithm genetic`
b. For running novelty search:
i. `python minimize.py --path test-suites/flask/tests --pop_size 50 --max_gen 100 --crossover 0.5 --mutation 0.2 --attribute_mutation 0.10 --tourn_size 3 --algorithm novelty`
c. For running novelty search with a different distance metric, or with the absence of the novelty archive:
i. `python minimize.py --path test-suites/flask/tests --pop_size 5 --max_gen 10 --crossover 0.99 --mutation 0.99 --attribute_mutation 0.99 --tourn_size 5 --algorithm novelty --distance_metric minkowski --omit_novelty_archive`
6. Observe that the test suite minimization has started with the appropriate parameters by checking the command line output.
7. Wait for the minimization to finish, this depends on the generation size that the command was run with, and the current CPU's performance.
8. Observe the output which details the minimized test suite size, and the reduction in code coverage percentage.
9. Open the `minimized_test_suite.csv` file in the same directory to observe a list of included test cases in the best test suite found.