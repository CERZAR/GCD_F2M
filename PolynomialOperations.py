import numpy as np
from common import P

# Алгоритм Евклида для НОД в поле Fp (p=2)
def calculate_gcd_sequential(polynomials):
    while len(polynomials) > 1:

        if len(polynomials) % 2 != 0:
            polynomials = np.vstack((polynomials, polynomials[-1]))

        polynomials_copy = polynomials.copy()
        results = np.array(polynomials[0:(len(polynomials)) // 2])
        results = results.copy()

        for i in range(len(polynomials) // 2):
            PolynomialOperations.process_polynomials_cpu(results, polynomials_copy, i)
        polynomials = results.copy()

    return polynomials[0]


class PolynomialOperations:

    @classmethod
    def modular_inverse_cpu(cls, n):
        x_0, x_1 = P, n
        s_0, s_1 = 0, 1
        i = 1

        while x_1 != 0:
            q = x_0 // x_1
            x_0, x_1 = x_1, x_0 - q * x_1
            s_0, s_1 = s_1, s_0 - q * s_1
            i += 1

        return s_0 % P

    @classmethod
    def polynomial_division_cpu(cls, f_copy, g, q):
        m = len(f_copy) - 1
        n = len(g) - 1

        deg_f = next((i for i in range(m, -1, -1) if f_copy[i]), 0)
        deg_g = next((i for i in range(n, -1, -1) if g[i]), 0)

        inv = g[deg_g]
        inverse_first_coef = cls.modular_inverse_cpu(inv)

        k = deg_f - deg_g

        while k >= 0:
            q[k] = (f_copy[deg_g + k] * inverse_first_coef) % P

            for j in range(deg_g + k, k - 1, -1):
                f_copy[j] = (f_copy[j] - q[k] * g[j - k]) % P
                if j == 0:
                    break

            if k == 0:
                break

            k -= 1

    @classmethod
    def polynomial_multiply_cpu(cls, a, b):
        deg_a = next((i for i in range(len(a) - 1, -1, -1) if a[i]), 0)
        deg_b = next((i for i in range(len(b) - 1, -1, -1) if b[i]), 0)

        result = [0] * (deg_a + deg_b + 1)

        for i in range(deg_a + 1):
            for j in range(deg_b + 1):
                result[i + j] = (result[i + j] + a[i] * b[j]) % P

        return result

    @classmethod
    def polynomial_difference_cpu(cls, a, b):
        min_len = min(len(a), len(b))
        diff = [(a[i] - b[i]) % P for i in range(min_len)]
        return diff

    @classmethod
    def set_nonzero_coefficients_cpu(cls, polynomial, not_null):
        not_null[0] = sum(polynomial)

    @classmethod
    def calculate_polynomial_gcd_cpu(cls, f, g, result, polynomial_id):
        m = len(f) - 1
        n = len(g) - 1
        p0, p1 = (f, g) if m >= n else (g, f)

        while any(p1):
            q = [0] * len(p0)
            p0_copy = p0.copy()

            cls.polynomial_division_cpu(p0_copy, p1, q)

            multiply = cls.polynomial_multiply_cpu(p1, q)
            difference = cls.polynomial_difference_cpu(p0, multiply)

            p0, p1 = p1, difference

            cls.set_nonzero_coefficients_cpu(p1, [0])

        result[polynomial_id][:len(p0)] = p0
        result[polynomial_id][len(p0):] = [0] * (len(result[polynomial_id]) - len(p0))

    @classmethod
    def process_polynomials_cpu(cls, res, polynomials_copy, polynomial_id):
        if polynomial_id < polynomials_copy.shape[0]:
            PolynomialOperations.calculate_polynomial_gcd_cpu(
                polynomials_copy[2 * polynomial_id],
                polynomials_copy[2 * polynomial_id + 1],
                res,
                polynomial_id
            )
