import subprocess
from common import DEGREE_MODE, AMOUNT_MODE, DEGREE_FILE, AMOUNT_FILE

SCRIPT_SEQUENTIAL = 'implementation_for_sequential.py'
SCRIPT_PARALLEL = 'implementation_for_cuda_parallel.py'
SCRIPT_PLOT_MAKER = 'gcd_algorithms_performance.py'

DEGREE_CONSTANTS = [(5000, 60)]
DEGREES = [10, 20, 30, 40, 50, 60, 70]
AMOUNTS = [10, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]


def run_command(script, degree, amount, mode=None):
    subprocess.run(['python', script,
                    '--degree', str(degree),
                    '--amount', str(amount),
                    '--mode', str(mode)
                    ])


def run_plot_maker(amount, degree):
    subprocess.run(['python', SCRIPT_PLOT_MAKER,
                    '--degree', str(degree),
                    '--amount', str(amount),
                    ])


def run_sequential_and_parallel(constants):
    clear_file(DEGREE_FILE)
    clear_file(AMOUNT_FILE)

    for degree in DEGREES:
        run_command(SCRIPT_SEQUENTIAL, degree, constants[0], DEGREE_MODE)
        run_command(SCRIPT_PARALLEL, degree, constants[0], DEGREE_MODE)

    for amount in AMOUNTS:
        run_command(SCRIPT_SEQUENTIAL, constants[1], amount, AMOUNT_MODE)
        run_command(SCRIPT_PARALLEL, constants[1], amount, AMOUNT_MODE)

    run_plot_maker(constants[0], constants[1])


def clear_file(file_path):
    with open(file_path, 'w'):
        pass


def main():
    for constants in DEGREE_CONSTANTS:
        run_sequential_and_parallel(constants)


if __name__ == "__main__":
    main()
