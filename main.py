import random
import re
import subprocess


def parse_and_build_test_case_data():
    build_test_case_cmd = "pytest --collect-only"

    build_test_case_output = subprocess.run(
        build_test_case_cmd, shell=True, capture_output=True, text=True
    )

    if build_test_case_output.returncode != 0:
        return False

    build_test_case_output = build_test_case_output.stdout

    test_suite_files = []
    test_cases_list = []

    # regexes
    regex_test_suite_file = "^<Module.* (.*.py)>$"
    regex_test_case = r"^  <Function (.*)\[.*>$"
    regex_test_class = "^  <Class (.*)>$"
    regex_test_class_test_case = r"^    <Function (.*)\[.*>$"

    current_test_suite_file = None
    current_test_class = None

    for line in build_test_case_output.splitlines():
        test_suit_file_result = re.search(regex_test_suite_file, line)

        if test_suit_file_result:
            test_suite_file = test_suit_file_result.group(1)
            test_suite_files.append(test_suite_file)
            current_test_suite_file = test_suite_file

        if current_test_suite_file:
            test_class_result = re.search(regex_test_class, line)

            if test_class_result:
                current_test_class = test_class_result.group(1)

            test_case_result = re.search(regex_test_case, line)

            test_class_test_case_result = re.search(regex_test_class_test_case, line)

            if test_class_test_case_result:
                test_cases_list.append(
                    "{}::{}::{}".format(
                        current_test_suite_file,
                        current_test_class,
                        test_class_test_case_result.group(1),
                    )
                )

            if test_case_result:
                test_cases_list.append(
                    "{}::{}".format(current_test_suite_file, test_case_result.group(1))
                )
                current_test_class = None

    return test_suite_files, test_cases_list


def run_custom_test_suite_and_calculate_test_coverage(
    test_cases_list=None, test_cases_activation_list=None
):
    test_cases_to_run = ""

    if test_cases_list and test_cases_activation_list:
        for test_case_activation in zip(test_cases_list, test_cases_activation_list):
            if test_case_activation[1]:
                test_cases_to_run += f" {test_case_activation[0]}"

    coverage_run_cmd = "coverage run --branch -m pytest{}".format(test_cases_to_run)

    print(coverage_run_cmd)

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
            statement_coverage_total,
            statement_coverage_missed,
            branch_coverage,
            branch_coverage_missed,
        )

    return False


def activate_random_test_suite(test_cases_list):
    return [random.choice([0, 1]) for x in test_cases_list]


# print(parse_and_build_test_case_data())
# print(calculate_total_coverage())

files, cases = parse_and_build_test_case_data()
print(files, cases)
random_activated_cases_list = activate_random_test_suite(cases)
print(random_activated_cases_list)
run_custom_test_suite_and_calculate_test_coverage(cases, random_activated_cases_list)
print(parse_coverage_report())
