import argparse
import subprocess

from polynomials_generation import GenerateDatasets

DEGREE_MODE = 'degrees'
AMOUNT_MODE = 'amount'
CONSOLE_MODE = 'console'
DEGREE_FILE = 'data/degrees.txt'
AMOUNT_FILE = 'data/amount.txt'
POLYNOMIALS_FILE = 'data/polynomials.txt'

THREADS_PER_BLOCK = (16, 16)
BLOCKS_PER_GRID = (32, 32)

P = 2
M = 4


def get_polynomials_from_file():
    polynomials = []
    with open(POLYNOMIALS_FILE, 'r') as file:
        for line in file:
            polynomial = eval(line.strip())
            polynomials.append(polynomial)
    return polynomials


def get_polynomials(implementation: str):
    description = 'Последовательный алгоритм' if implementation == 's' else 'Параллельный алгоритм'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--degree', type=int, help='Максимальная степень (m) полиномов в поле F2^m')
    parser.add_argument('--amount', type=int, help='Количество полиномов в поле F2^m')
    parser.add_argument('--mode', type=str, help='Режим записи логов (amount - amount.txt | degrees - degrees.txt)')
    args = parser.parse_args()

    if args.degree is None or args.amount is None:
        if implementation == 's':
            subprocess.run(['python', 'polynomials_generation.py'])
        # subprocess.run(['python', 'polynomials_generation.py'])
        args.mode = CONSOLE_MODE
    else:
        GenerateDatasets.generate_dataset(args.degree, args.amount)

    polynomials = get_polynomials_from_file()

    polynomial_degree = len(polynomials[0])
    polynomials_amount = len(polynomials)

    return polynomials, polynomial_degree, polynomials_amount, args.mode


def log_results(mode, performance_results, gcd):
    if mode == AMOUNT_MODE:
        with open(AMOUNT_FILE, 'a') as file:
            file.write(f'{performance_results}\n')
    elif mode == DEGREE_MODE:
        with open(DEGREE_FILE, 'a') as file:
            file.write(f'{performance_results}\n')
    elif mode == CONSOLE_MODE:
        implementation = 'последовательная на CPU' if performance_results[0] == 's' else 'параллельная на GPU'
        print("Реализация алгоритма: ", implementation)
        print("Степень полиномов: ", performance_results[1])
        print("Количество полиномов: ", performance_results[2])
        print("Время работы: ", performance_results[3])
        print("НОД: ", gcd)
