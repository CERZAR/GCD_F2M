from typing import List

from numba import cuda, float64
import numpy as np
import time
from common import get_polynomials, log_results, P


@cuda.jit
def modular_inverse(n):
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
def polynomial_division(f_copy, g, q):
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
    inverse_first_coef = modular_inverse(inv)
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
def polynomial_multiply(a, b, result):
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
def polynomial_difference(a, b, diff):
    for i in range(min(len(a), len(b))):
        diff[i] = (a[i] - b[i]) % P


@cuda.jit
def set_nonzero_coefficients(polynomial, not_null):
    element_sum = 0
    for i in range(len(polynomial)):
        element_sum += polynomial[i]
    not_null[0] = element_sum


@cuda.jit
def calculate_polynomial_gcd(f, g, gcd, polynomial_id):
    shape = 100
    m = len(f) - 1
    n = len(g) - 1
    p0, p1 = (f, g) if m >= n else (g, f)

    not_null = cuda.local.array(1, np.float64)
    p0_copy = cuda.local.array(shape, np.float64)
    multiply = cuda.local.array(shape, float64)
    difference = cuda.local.array(shape, float64)
    q = cuda.local.array(shape, np.float64)
    set_nonzero_coefficients(p1, not_null)

    while not_null[0]:
        for j in range(len(p0)):
            q[j] = 0
        for j in range(len(p0)):
            p0_copy[j] = p0[j]
        polynomial_division(p0_copy, p1, q)
        for j in range(len(p0)):
            multiply[j] = 0
        for j in range(len(p0)):
            difference[j] = 0
        polynomial_multiply(p1, q, multiply)
        polynomial_difference(p0, multiply, difference)
        for j in range(len(p0)):
            p0[j] = p1[j]
        for j in range(len(p0)):
            p1[j] = difference[j]
        set_nonzero_coefficients(p1, not_null)

    for i in range(len(p0)):
        gcd[polynomial_id][i] = p0[i]
    i = len(p0)

    while i < len(gcd[polynomial_id]):
        gcd[polynomial_id][i] = 0
        i += 1


@cuda.jit
def process_polynomials(res, polynomials_copy):
    polynomial_id = cuda.grid(1)
    if polynomial_id < polynomials_copy.shape[1]:
        calculate_polynomial_gcd(
            polynomials_copy[2 * polynomial_id],
            polynomials_copy[2 * polynomial_id + 1],
            res,
            polynomial_id
        )


def reduction_polynomials(polynomials: List[List[int]]) -> List[int]:
    coef_max_degree = len(polynomials[0])
    poly_max_degree = len(polynomials) - 1
    result = [0] * (coef_max_degree + poly_max_degree)

    for i, poly in enumerate(polynomials):
        for j, value in enumerate(poly):
            pos = i + j
            result[pos] = (result[pos] + value) % 2

    return result


@cuda.jit
def reduction_polynomials_kernel(polynomials, result):
    i, j = cuda.grid(2)
    if i < polynomials.shape[0] and j < polynomials.shape[1]:
        pos = i + j
        result[pos] = (result[pos] + polynomials[i, j]) % 2


def reduction_polynomials_parallel(polynomials):
    polynomials_np = np.array(polynomials, dtype=np.int32)
    coef_max_degree = polynomials_np.shape[1]
    poly_max_degree = polynomials_np.shape[0] - 1
    result = np.zeros(coef_max_degree + poly_max_degree, dtype=np.int32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (polynomials_np.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (polynomials_np.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    reduction_polynomials_kernel[blocks_per_grid, threads_per_block](polynomials_np, result)
    return result


def main():
    implementation = 'p'
    polynomials, polynomial_degree, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    # polynomials = [reduction_polynomials(poly) for poly in polynomials]
    # polynomials = np.array(polynomials)
    polynomials = np.array([reduction_polynomials_parallel(poly) for poly in polynomials])

    while len(polynomials) > 1:

        if len(polynomials) % 2 != 0:
            polynomials = np.vstack((polynomials, polynomials[-1]))

        cudapolynomials = cuda.to_device(polynomials)
        results = np.array(polynomials[0:(len(polynomials)) // 2])
        results = cuda.to_device(results)
        process_polynomials.forall(len(polynomials) // 2)(results, cudapolynomials)
        polynomials = results.copy_to_host()

    work_time = time.time() - start_time

    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time]
    log_results(mode, result_tests, polynomials[0])


if __name__ == "__main__":
    main()
