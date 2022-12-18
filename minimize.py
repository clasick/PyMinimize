import argparse
import random
import re
import subprocess
import os

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

if __name__ == "__main__":
    # parse the command line arguments for the program
    parser = argparse.ArgumentParser(
            description="Parse arguments for minimize.py")
    parser.add_argument('test_suite_path', metavar="test_suite_path", 
            type=str, 
            help="Path to directory containing test suite")
    # parser.add_argument('--temp', metavar="temperature", 
    #         type=float,
    #         default = 1000,
    #         required=False,
    #         help="temperature for performing SA algorithm")
    # parser.add_argument('--cool', metavar="cooling_factor", 
    #         type=float,
    #         default = 0.9999,
    #         required = False,
    #         help="cooling factor for performing SA algorithm")

    args = parser.parse_args()
    test_suite_path = args.test_suite_path
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
