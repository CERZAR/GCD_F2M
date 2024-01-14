from random import randint
import argparse


def get_random_polynomial(m, p):
    polynomial = []
    for _ in range(m + 1):
        polynomial.append(randint(0, p - 1))
    return polynomial


class GenerateDatasets:
    """
    Класс для генерации наборов полиномов в поле Fp^m.
    """

    @classmethod
    def generate_dataset(cls, m, polynomial_degree, polynomials_amount):
        """
        Генерирует набор полиномов и записывает их в файл.

        Args:
            m (int): Размерность поля F2^m
            polynomial_degree (int): Степень генерируемых полиномов.
            polynomials_amount (int): Количество генерируемых полиномов.

        Returns:
            None

        Сгенерированный набор записывается в файл с именем 'polynomials.txt'.
        Каждая строка в файле представляет собой полином в виде списка коэффициентов.
        """
        from common import POLYNOMIALS_FILE, P
        dataset = []

        # Предварительная генерация полинома в поле F2^m.
        # Здесь polynomial - массив со степенями alpha в полиноме.
        # Сам полином имеет вид alpha*x^n + alpha*x^n-1 + ... + alpha*x + alpha
        for i in range(polynomials_amount):
            polynomial = get_random_polynomial(polynomial_degree, P ** m)
            polynomial[-1] = randint(1, P ** m - 1)
            dataset.append(polynomial)

        with open(POLYNOMIALS_FILE, 'w') as file:
            file.write(f'm: {m}\n')
            for polynomial in dataset:
                file.write(f'{polynomial}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Генерация полиномов в поле F2^m')
    parser.add_argument('--m', type=int, help='Степень поля F2^m')
    parser.add_argument('--degree', type=int, help='Максимальная степень полиномов')
    parser.add_argument('--amount', type=int, help='Количество полиномов')
    args = parser.parse_args()

    if args.m is None:
        args.m = int(input('Введите степень поля F2^m: '))

    if args.degree is None:
        args.degree = int(input('Введите максимальную степень полиномов: '))

    if args.amount is None:
        args.amount = int(input('Введите количество полиномов: '))

    GenerateDatasets.generate_dataset(
        m=args.m,
        polynomial_degree=args.degree,
        polynomials_amount=args.amount
    )
