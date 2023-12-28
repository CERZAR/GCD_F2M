from random import randint
import argparse


class GenerateDatasets:
    """
    Класс для генерации наборов полиномов в поле Fp^m.
    """

    @classmethod
    def generate_dataset(cls, polynomial_degree, polynomials_amount):
        """
        Генерирует набор полиномов и записывает их в файл.

        Args:
            polynomial_degree (int): Степень генерируемых полиномов.
            polynomials_amount (int): Количество генерируемых полиномов.

        Returns:
            None

        Сгенерированный набор записывается в файл с именем 'polynomials.txt'.
        Каждая строка в файле представляет собой полином в виде списка коэффициентов.
        """
        from common import POLYNOMIALS_FILE, P
        dataset = []

        for i in range(polynomials_amount):
            polynomial = []
            for j in range(polynomial_degree):
                degree = []
                for _ in range(polynomial_degree):
                    degree.append(randint(0, P - 1))
                polynomial.append(degree)
            dataset.append(polynomial)

        with open(POLYNOMIALS_FILE, 'w') as file:
            for polynomial in dataset:
                file.write(f'{polynomial}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Генерация полиномов в поле F2^m')
    parser.add_argument('--degree', type=int, help='Максимальная степень (m) полиномов в поле F2^m')
    parser.add_argument('--amount', type=int, help='Количество полиномов в поле F2^m')
    args = parser.parse_args()

    if args.degree is None:
        args.degree = int(input('Введите степень (m) полиномов в поле F2^m: '))

    if args.amount is None:
        args.amount = int(input('Введите количество полиномов: '))

    GenerateDatasets.generate_dataset(
        polynomial_degree=args.degree,
        polynomials_amount=args.amount
    )
