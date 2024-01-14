from random import randint
import numpy as np
from common import P


class FieldBuilder:

    def set_up(self):
        all_underfields_coenfs = FieldBuilder.get_underfields_coefs(self.m)
        for underfield_coef in all_underfields_coenfs:
            self.underfield_polynomials.append(FieldBuilder.get_underfield_poly(underfield_coef))

    def __init__(self, m):
        self.underfield_polynomials = []
        self.m = m
        self.p = P
        self.set_up()

    @classmethod
    def get_random_poly(cls, m):
        polynomial = []
        for j in range(m + 1):
            polynomial.append(randint(0, 1))
        return polynomial

    @classmethod
    def get_underfield_poly(cls, underfield_coef):
        poly_degree = P ** underfield_coef
        polynomial = [0] * (poly_degree + 1)
        polynomial[1] = 1
        polynomial[-1] = 1
        return polynomial

    @classmethod
    def prime_factors(cls, n):
        factors = []
        divisor = 2
        while n > 1:
            while n % divisor == 0:
                factors.append(divisor)
                n //= divisor
            divisor += 1
        return factors

    @classmethod
    def prime_factors_for_composite_number(cls, num):
        """
        Поиск всех простых подмножителей числа (где число - это m из F2^m)
        """
        result = []
        for i in range(2, num + 1):
            if num % i == 0:
                factors = cls.prime_factors(i)
                result.extend(factors)
        return list(set(result))

    @classmethod
    def get_underfields_coefs(cls, m):
        """
        Вычисление степеней подполей поля F2^m
        """
        prime_numbers = cls.prime_factors_for_composite_number(m)
        underfields_coefs = []

        for prime in prime_numbers:
            underfields_coefs.append(int(m / prime))
        return underfields_coefs

    @staticmethod
    def is_single_polynomial(polynomial: list) -> bool:
        if polynomial[0] != 1:
            return False

        if any(x != 0 for x in polynomial[1:]):
            return False

        return True

    def is_irreducible_polynomial(self, f):
        from PolynomialOperations import calculate_gcd_sequential

        if FieldBuilder.is_single_polynomial(f):
            return False

        if f[-1] == 0:
            return False

        for underfield_polynomial in self.underfield_polynomials:
            max_poly_length = max(len(f), len(underfield_polynomial))

            if len(f) != max_poly_length:
                f = f + [0] * (max_poly_length - len(f))
            else:
                underfield_polynomial = underfield_polynomial + [0] * (max_poly_length - len(underfield_polynomial))

            polynomials = np.array([f, underfield_polynomial])
            gcd = calculate_gcd_sequential(polynomials)

            if not FieldBuilder.is_single_polynomial(gcd):
                return False

        return True

    def decimal_to_binary(self, decimal_number):
        binary_array = []

        while decimal_number > 0:
            remainder = decimal_number % 2
            binary_array.insert(0, remainder)
            decimal_number //= 2

        while len(binary_array) < self.m + 1:
            binary_array.insert(0, 0)

        return binary_array

    def calculate_irreducible_polynomial(self):
        """
        Поиск неприводимого полинома над полем F2^m
        """
        max_number = (2 ** (self.m + 1)) - 1
        for number in range(1, max_number + 1):
            polynomial = self.decimal_to_binary(number)
            temp_res = self.is_irreducible_polynomial(polynomial)
            if temp_res:
                return polynomial
        raise Exception('incorrect dataset')

    def calculate_alphas_dict(self, irreducible_polynomial):
        """
        Вычисление полиномов для alpha относительно неприводимого полинома поля F2^m
        Количество возможных alpha - 2^m
        Максимальная степень полинома - m

        На выходе - словарь значений alpha (0: 0,
                                            1: alpha^0,
                                            2: alpha^1,
                                            ...,
                                            2^m-1: alpha^m)
        """
        alpha_max_degree = 2 ** self.m
        alpha_max_value = irreducible_polynomial.copy()
        alpha_max_value[-1] = 0

        alpha_dict = {
            0: [0] * (self.m + 1),
            1: [1] + [0] * self.m,
        }

        for alpha_degree in range(2, alpha_max_degree):
            temp_poly = alpha_dict[alpha_degree - 1][:]

            # Умножили
            temp_poly = [0] + temp_poly[:-1]
            temp_poly[-1] ^= alpha_dict[alpha_degree - 1][-1]

            # Подставили max_alpha_degree
            if temp_poly[-1]:
                temp_poly = [a ^ b for a, b in zip(temp_poly, alpha_max_value)]
                temp_poly[-1] = 0

            alpha_dict[alpha_degree] = temp_poly

        for poly in alpha_dict.values():
            poly.pop()

        return alpha_dict
