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


def get_polynomials_from_file():
    polynomials = []
    with open(POLYNOMIALS_FILE, 'r') as file:
        first_line = file.readline()
        m = int(first_line.split(":")[1].strip())  # Extract the integer value after "m:"

        for line in file:
            polynomial = eval(line.strip())
            polynomials.append(polynomial)
    return m, polynomials


def get_polynomials(implementation: str):
    description = 'Последовательный алгоритм' if implementation == 's' else 'Параллельный алгоритм'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--m', type=int, help='Степень поля F2^m')
    parser.add_argument('--degree', type=int, help='Максимальная степень полиномов')
    parser.add_argument('--amount', type=int, help='Количество полиномов')
    parser.add_argument('--mode', type=str, help='Режим записи логов (amount - amount.txt | degrees - degrees.txt)')
    args = parser.parse_args()

    if args.m is None or args.degree is None or args.amount is None:
        subprocess.run(['python', 'polynomials_generation.py'])
        args.mode = CONSOLE_MODE
    else:
        GenerateDatasets.generate_dataset(args.m, args.degree, args.amount)

    m, polynomials = get_polynomials_from_file()

    polynomial_degree = len(polynomials[0])
    polynomials_amount = len(polynomials)

    return polynomials, m, polynomials_amount, args.mode


def log_results(mode, performance_results, gcd):
    if mode == AMOUNT_MODE:
        with open(AMOUNT_FILE, 'a') as file:
            file.write(f'{performance_results}\n')
    elif mode == DEGREE_MODE:
        with open(DEGREE_FILE, 'a') as file:
            file.write(f'{performance_results}\n')
    elif mode == CONSOLE_MODE:
        implementation = 'последовательная на CPU' if performance_results[0] == 's' else 'параллельная на GPU'
        print(f"Реализация алгоритма в поле F2^{performance_results[4]}: {implementation}")
        print("Степень полиномов: ", performance_results[1])
        print("Количество полиномов: ", performance_results[2])
        print("Время работы: ", performance_results[3])
        print("НОД: ", gcd)
