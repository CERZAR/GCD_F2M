import time
from typing import List

from FieldBuilder import FieldBuilder
from common import get_polynomials, log_results


# Алгоритм Евклида для НОД полиномов в поле F2^m
class PolynomialOperationsF2mCPU:

    def set_up(self):
        self.p = 2 ** self.m - 1

        if not self.alphas_dict:
            fb = FieldBuilder(self.m)
            irreducible_polynomial = fb.calculate_irreducible_polynomial()
            alphas_dict = fb.calculate_alphas_dict(irreducible_polynomial)
            self.alphas_dict = alphas_dict

    def __init__(self, m, alphas_dict=None):
        self.p = None
        self.alphas_dict = alphas_dict
        self.m = m
        self.set_up()

    def alpha_mod(self, alpha: int) -> int:
        """
        Кастомный мод для alpha. Индексация alpha начинается с 1
        """
        if alpha == 0:
            return 0
        return (alpha - 1) % self.p + 1

    def alpha_modular_inverse(self, a: int, b: int) -> int:
        """
        Находит значение x для уравнения a + x ≡ b (mod p),
        где a, b, и p - целые числа.
        """
        # при умножении альф нельзя получить 0. alpha^0 имеет индекс 1.
        return self.alpha_mod(b - a) + 1

    def alpha_multiply(self, alpha1: int, alpha2: int) -> int:
        # при умножении альф нельзя получить 0. alpha^0 имеет индекс 1.
        return self.alpha_mod(alpha1 + alpha2 - 1)

    def alpha_summary(self, alpha1: int, alpha2: int) -> int:
        new_alhpa_value = [a ^ b for a, b in zip(self.alphas_dict[alpha1], self.alphas_dict[alpha2])]
        for key, val in enumerate(self.alphas_dict):
            if val == new_alhpa_value:
                return key
        return 0

    def get_max_degree(self, poly: List[int]) -> int:
        max_degree = 0
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                max_degree = i
                break
        return max_degree

    def get_multiplier(self, poly_f: List[int], poly_g: List[int]) -> List[int]:
        max_degree_f = self.get_max_degree(poly_f)
        max_degree_g = self.get_max_degree(poly_g)

        multiplier_degree = abs(max_degree_f - max_degree_g)
        multiplier_value = self.alpha_modular_inverse(poly_g[max_degree_g], poly_f[max_degree_f])

        multiplier_poly = [0] * len(poly_f)
        multiplier_poly[multiplier_degree] = multiplier_value
        return multiplier_poly

    def polynomial_multiply(self, poly_f: List[int], poly_g: List[int]) -> List[int]:
        poly_f_max_degree = self.get_max_degree(poly_f)
        poly_g_max_degree = self.get_max_degree(poly_g)

        List_size = max((poly_f_max_degree + poly_g_max_degree + 1), max(len(poly_f), len(poly_g)))
        result = [0] * List_size

        for i, f_value in reversed(list(enumerate(poly_f))):
            if f_value:
                for j, g_value in enumerate(poly_g):
                    if g_value:
                        pos = i + j
                        temp_value = self.alpha_multiply(f_value, g_value)
                        result[pos] = self.alpha_summary(result[pos], temp_value)

        return result

    def polynomial_difference(self, poly_f: List[int], poly_g: List[int]) -> List[int]:
        result = [0] * len(poly_f)
        for i in range(len(poly_f)):
            result[i] = self.alpha_summary(poly_f[i], poly_g[i])
        return result

    def polynomial_division(self, poly_f: List[int], poly_g: List[int]) -> List[int]:
        """
        f / g
        Возвращает остаток от деления, без целой части
        """
        divisible = poly_f
        divisor = poly_g

        while True:
            max_degree_divisible = self.get_max_degree(divisible)
            max_degree_divisor = self.get_max_degree(divisor)

            if max_degree_divisible < max_degree_divisor:
                break

            multiplier = self.get_multiplier(divisible, divisor)
            temp_poly = self.polynomial_multiply(divisor, multiplier)
            divisible = self.polynomial_difference(divisible, temp_poly)

            if self.is_single_polynomial(divisible):
                break

        return divisible

    def is_single_polynomial(self, poly: List[int]) -> bool:
        if any(x != 0 for x in poly):
            return False
        return True

    def polynomial_GCD(self, poly_f: List[int], poly_g: List[int]) -> List[int]:
        max_degree_poly_f = self.get_max_degree(poly_f)
        max_degree_poly_g = self.get_max_degree(poly_g)

        if max_degree_poly_f > max_degree_poly_g:
            divisible = poly_f.copy()
            divisor = poly_g.copy()
        else:
            divisible = poly_g.copy()
            divisor = poly_f.copy()

        while True:
            left_over = self.polynomial_division(divisible, divisor)
            if self.is_single_polynomial(left_over):
                return divisor
            divisor, divisible = left_over, divisor


def calculate_gcd_sequential(poly_list: List[List[int]], m: int) -> List[int]:
    po = PolynomialOperationsF2mCPU(m)
    gcd_result = po.polynomial_GCD(poly_list[0], poly_list[1])

    for poly in poly_list[2:]:
        gcd_result = po.polynomial_GCD(poly, gcd_result)

    return gcd_result


def main():
    implementation = 's'
    polynomials, m, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    gcd = calculate_gcd_sequential(polynomials, m)

    work_time = time.time() - start_time

    polynomial_degree = len(gcd) - 1
    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time, m]
    log_results(mode, result_tests, gcd)


if __name__ == "__main__":
    main()
