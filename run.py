import subprocess
from common import DEGREE_MODE, AMOUNT_MODE, DEGREE_FILE, AMOUNT_FILE

SCRIPT_SEQUENTIAL = 'PolynomialOperationsF2mCPU.py'
SCRIPT_PARALLEL = 'PolynomialOperationsF2mGPU.py'
SCRIPT_PLOT_MAKER = 'gcd_algorithms_performance.py'

CONSTANTS = [(400, 20, 10)]
DEGREES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
AMOUNTS = [10, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]


def run_command(script, m, degree, amount, mode=None):
    subprocess.run(['python', script,
                    '--m', str(m),
                    '--degree', str(degree),
                    '--amount', str(amount),
                    '--mode', str(mode)
                    ])


def run_plot_maker(m, amount, degree):
    subprocess.run(['python', SCRIPT_PLOT_MAKER,
                    '--m', str(m),
                    '--degree', str(degree),
                    '--amount', str(amount),
                    ])


def run_sequential_and_parallel(constants):
    clear_file(DEGREE_FILE)
    clear_file(AMOUNT_FILE)

    for degree in DEGREES:
        run_command(SCRIPT_SEQUENTIAL, constants[2], degree, constants[0], DEGREE_MODE)
        run_command(SCRIPT_PARALLEL, constants[2], degree, constants[0], DEGREE_MODE)

    for amount in AMOUNTS:
        run_command(SCRIPT_SEQUENTIAL, constants[2], constants[1], amount, AMOUNT_MODE)
        run_command(SCRIPT_PARALLEL, constants[2], constants[1], amount, AMOUNT_MODE)

    run_plot_maker(constants[2], constants[0], constants[1])


def clear_file(file_path):
    with open(file_path, 'w'):
        pass


def main():
    for constants in CONSTANTS:
        run_sequential_and_parallel(constants)


if __name__ == "__main__":
    main()
