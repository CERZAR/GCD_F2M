from typing import List
import numpy as np
from numba import cuda, float64

from common import P, BLOCKS_PER_GRID, THREADS_PER_BLOCK


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

    @staticmethod
    @cuda.jit
    def polynomial_reduction_cuda_kernel(polynomials, result):
        i, j = cuda.grid(2)

        if i < len(polynomials) and j < len(polynomials[0]):
            pos = i + j
            cuda.atomic.add(result, pos, polynomials[i, j])

    @classmethod
    def polynomial_reduction_gpu(cls, polynomials: List[List[int]]) -> List[int]:
        polynomials_np = np.array(polynomials, dtype=np.int32)

        coef_max_degree = len(polynomials_np[0])
        poly_max_degree = len(polynomials_np) - 1
        result = np.zeros(coef_max_degree + poly_max_degree, dtype=np.int32)

        polynomials_gpu = cuda.to_device(polynomials_np)
        result_gpu = cuda.to_device(result)

        PolynomialOperations.polynomial_reduction_cuda_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](polynomials_gpu,
                                                                                                  result_gpu)

        result_gpu.copy_to_host(result)

        return np.mod(result, 2).tolist()

    @classmethod
    def polynomial_reduction_cpu(cls, polynomials: List[List[int]]) -> List[int]:
        coef_max_degree = len(polynomials[0])
        poly_max_degree = len(polynomials) - 1
        result = [0] * (coef_max_degree + poly_max_degree)

        for i, poly in enumerate(polynomials):
            for j, value in enumerate(poly):
                pos = i + j
                result[pos] = (result[pos] + value) % P

        return result

    @staticmethod
    @cuda.jit
    def process_polynomials_gpu(res, polynomials_copy):
        polynomial_id = cuda.grid(1)
        if polynomial_id < polynomials_copy.shape[1]:
            calculate_polynomial_gcd_gpu(
                polynomials_copy[2 * polynomial_id],
                polynomials_copy[2 * polynomial_id + 1],
                res,
                polynomial_id
            )


@cuda.jit
def calculate_polynomial_gcd_gpu(f, g, gcd, polynomial_id):
    shape = 2000
    m = len(f) - 1
    n = len(g) - 1
    p0, p1 = (f, g) if m >= n else (g, f)

    not_null = cuda.local.array(1, np.float64)
    p0_copy = cuda.local.array(shape, np.float64)
    multiply = cuda.local.array(shape, float64)
    difference = cuda.local.array(shape, float64)
    q = cuda.local.array(shape, np.float64)
    set_nonzero_coefficients_gpu(p1, not_null)

    while not_null[0]:
        for j in range(len(p0)):
            q[j] = 0
        for j in range(len(p0)):
            p0_copy[j] = p0[j]
        polynomial_division_gpu(p0_copy, p1, q)
        for j in range(len(p0)):
            multiply[j] = 0
        for j in range(len(p0)):
            difference[j] = 0
        polynomial_multiply_gpu(p1, q, multiply)
        polynomial_difference_gpu(p0, multiply, difference)
        for j in range(len(p0)):
            p0[j] = p1[j]
        for j in range(len(p0)):
            p1[j] = difference[j]
        set_nonzero_coefficients_gpu(p1, not_null)

    for i in range(len(p0)):
        gcd[polynomial_id][i] = p0[i]
    i = len(p0)

    while i < len(gcd[polynomial_id]):
        gcd[polynomial_id][i] = 0
        i += 1


@cuda.jit
def modular_inverse_gpu(n):
    x_0, x_1 = P, n
    s_0, s_1 = 0, 1
    i = 1

    while x_1 != 0:
        q = x_0 // x_1
        x_0, x_1 = x_1, x_0 - q * x_1
        s_0, s_1 = s_1, s_0 - q * s_1
        i += 1

    return s_0 % P


@cuda.jit
def polynomial_division_gpu(f_copy, g, q):
    m = len(f_copy) - 1
    n = len(g) - 1

    deg_f = 0
    for i in range(m + 1):
        if f_copy[i]:
            deg_f = i
    deg_g = 0
    for i in range(n + 1):
        if g[i]:
            deg_g = i

    inv = g[deg_g]
    inverse_first_coef = modular_inverse_gpu(inv)
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


@cuda.jit
def polynomial_multiply_gpu(a, b, result):
    deg_a = 0
    deg_b = 0

    for i, val_a in enumerate(a):
        if val_a:
            deg_a = i

    for i, val_b in enumerate(b):
        if val_b:
            deg_b = i

    for i, val_a in enumerate(a[:deg_a + 1]):
        for j, val_b in enumerate(b[:deg_b + 1]):
            result[i + j] += val_a * val_b % P
            result[i + j] %= P


@cuda.jit
def polynomial_difference_gpu(a, b, diff):
    for i in range(min(len(a), len(b))):
        diff[i] = (a[i] - b[i]) % P


@cuda.jit
def set_nonzero_coefficients_gpu(polynomial, not_null):
    element_sum = 0
    for i in range(len(polynomial)):
        element_sum += polynomial[i]
    not_null[0] = element_sum
