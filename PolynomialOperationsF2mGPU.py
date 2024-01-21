import time
import numpy as np
from FieldBuilder import FieldBuilder
from common import get_polynomials, log_results
from numba import cuda


# Алгоритм Евклида для НОД полиномов в поле F2^m
@cuda.jit
def alpha_mod(alpha, p, result):
    i = cuda.grid(1)
    if i < len(alpha):
        result[i] = (alpha[i] - 1) % p + 1


@cuda.jit
def alpha_modular_inverse(a, b, p, result):
    i = cuda.grid(1)
    if i < len(a):
        # при умножении альф нельзя получить 0. alpha^0 имеет индекс 1.
        result[i] = alpha_mod((b[i] - a[i]) % p, p) + 1


@cuda.jit
def alpha_multiply(alpha1, alpha2, p, result):
    i = cuda.grid(1)
    if i < len(alpha1):
        # при умножении альф нельзя получить 0. alpha^0 имеет индекс 1.
        result[i] = alpha_mod((alpha1[i] + alpha2[i] - 1) % p, p)


@cuda.jit
def alpha_summary(alpha1, alpha2, alpha_dict, result):
    i = cuda.grid(1)
    if i < len(alpha_dict):
        new_alpha_value = [a ^ b for a, b in zip(alpha_dict[alpha1][i], alpha_dict[alpha2][i])]
        for key, val in enumerate(alpha_dict):
            if val == new_alpha_value:
                result[i] = key
                return
        result[i] = -1


@cuda.jit
def get_max_degree(poly, result):
    i = cuda.grid(1)
    if i < len(poly):
        if poly[i] != 0:
            result[0] = i


@cuda.jit
def get_multiplier(poly_f, poly_g, p, result):
    max_degree_f = get_max_degree(poly_f)
    max_degree_g = get_max_degree(poly_g)

    multiplier_degree = abs(max_degree_f - max_degree_g)
    multiplier_value = alpha_modular_inverse(poly_g[max_degree_g], poly_f[max_degree_f], p)

    result[multiplier_degree] = multiplier_value


@cuda.jit
def polynomial_multiply_kernel(poly_f, poly_g, p, alpha_dict, result):
    i, j = cuda.grid(2)
    if i < len(poly_f) and j < len(poly_g):
        if poly_f[i] and poly_g[j]:
            pos = i + j
            temp_value = alpha_multiply(poly_f[i], poly_g[j], p)
            result[pos] = alpha_summary(result[pos], temp_value, alpha_dict)


def polynomial_multiply(poly_f, poly_g, p, alpha_dict):
    poly_f_max_degree = get_max_degree(poly_f)
    poly_g_max_degree = get_max_degree(poly_g)

    list_size = max((poly_f_max_degree + poly_g_max_degree + 1), max(len(poly_f), len(poly_g)))
    result = [0] * list_size

    d_poly_f = cuda.to_device(poly_f)
    d_poly_g = cuda.to_device(poly_g)
    d_result = cuda.to_device(result)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (len(poly_f) + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (len(poly_g) + threads_per_block[1] - 1) // threads_per_block[1]

    polynomial_multiply_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
        d_poly_f, d_poly_g, p, alpha_dict, d_result
    )

    cuda.synchronize()
    d_result.copy_to_host(result)

    return result


@cuda.jit
def polynomial_difference_kernel(poly_f, poly_g, alpha_dict, result):
    i = cuda.grid(1)
    if i < len(poly_f):
        result[i] = alpha_summary(poly_f[i], poly_g[i], alpha_dict)


def polynomial_difference(poly_f, poly_g, alpha_dict):
    result = [0] * len(poly_f)

    d_poly_f = cuda.to_device(poly_f)
    d_poly_g = cuda.to_device(poly_g)
    d_result = cuda.to_device(result)

    threads_per_block = 256
    blocks_per_grid = (len(poly_f) + threads_per_block - 1) // threads_per_block

    polynomial_difference_kernel[blocks_per_grid, threads_per_block](
        d_poly_f, d_poly_g, alpha_dict, d_result
    )

    cuda.synchronize()
    d_result.copy_to_host(result)

    return result


@cuda.jit
def polynomial_division_kernel(divisible, divisor, p, alpha_dict, result):
    max_degree_divisible = get_max_degree(divisible)
    max_degree_divisor = get_max_degree(divisor)

    while max_degree_divisible >= max_degree_divisor:
        multiplier = get_multiplier(divisible, divisor, p, alpha_dict)
        temp_poly = polynomial_multiply(divisor, multiplier, p, alpha_dict)
        divisible = polynomial_difference(divisible, temp_poly, alpha_dict)

        max_degree_divisible = get_max_degree(divisible)

    for i in range(len(divisible)):
        result[i] = divisible[i]


def polynomial_division(poly_f, poly_g, p, alpha_dict):
    d_divisible = cuda.to_device(poly_f)
    d_divisor = cuda.to_device(poly_g)
    d_result = cuda.to_device([0] * len(poly_f))

    threads_per_block = 256
    blocks_per_grid = (len(poly_f) + threads_per_block - 1) // threads_per_block

    polynomial_division_kernel[blocks_per_grid, threads_per_block](
        d_divisible, d_divisor, p, alpha_dict, d_result
    )

    cuda.synchronize()
    result = d_result.copy_to_host()

    return result


@cuda.jit
def is_single_polynomial_kernel(poly, result):
    result[0] = 1 if all(x == 0 for x in poly) else 0


@cuda.jit
def polynomial_GCD_kernel(poly_f, poly_g, p, alpha_dict, result, polynomial_id):
    divisible = poly_f
    divisor = poly_g
    left_over = divisible

    while True:
        polynomial_division_kernel(divisible, divisor, p, alpha_dict, left_over)
        is_single = is_single_polynomial_kernel(left_over)

        if is_single:
            break

        divisor, divisible = left_over, divisor

    for i in range(len(divisor)):
        result[polynomial_id][i] = divisor[i]

    i = len(divisor)

    while i < len(result[polynomial_id]):
        result[polynomial_id][i] = 0
        i += 1


@cuda.jit
def is_single_polynomial(poly):
    result = cuda.to_device([0])
    is_single_polynomial_kernel[poly.size, 1](poly, result)
    return result.copy_to_host()[0]


@cuda.jit
def polynomial_GCD(poly_f, poly_g, p, alpha_dict, result, polynomial_id):
    d_poly_f = cuda.to_device(poly_f)
    d_poly_g = cuda.to_device(poly_g)
    d_result = cuda.to_device(result)

    threads_per_block = 256
    blocks_per_grid = (len(poly_f) + threads_per_block - 1) // threads_per_block

    polynomial_GCD_kernel[blocks_per_grid, threads_per_block](
        d_poly_f, d_poly_g, p, alpha_dict, d_result, polynomial_id
    )

    cuda.synchronize()
    result = d_result.copy_to_host()

    return result


@cuda.jit
def process_polynomials_gpu(res, polynomials_copy, p, alpha_dict):
    polynomial_id = cuda.grid(1)
    if polynomial_id < polynomials_copy.shape[1]:
        polynomial_GCD(
            polynomials_copy[2 * polynomial_id],
            polynomials_copy[2 * polynomial_id + 1],
            p,
            alpha_dict,
            res,
            polynomial_id
        )


def calculate_gcd_parallel(polynomials, m):
    p = 2 ** m - 1
    fb = FieldBuilder(m)
    irreducible_polynomial = fb.calculate_irreducible_polynomial()
    alphas_dict = fb.calculate_alphas_dict(irreducible_polynomial)
    cuda_alphas_dict = cuda.to_device(alphas_dict)

    while len(polynomials) > 1:

        if len(polynomials) % 2 != 0:
            polynomials = np.vstack((polynomials, polynomials[-1]))

        cuda_polynomials = cuda.to_device(polynomials)
        results = np.array(polynomials[0:(len(polynomials)) // 2])
        results = cuda.to_device(results)
        process_polynomials_gpu.forall(len(polynomials) // 2)(results, cuda_polynomials, p, cuda_alphas_dict)
        polynomials = results.copy_to_host()

    return polynomials[0]


def main():
    implementation = 'p'
    polynomials, m, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    gcd = calculate_gcd_parallel(polynomials, m)

    work_time = time.time() - start_time

    polynomial_degree = len(gcd) - 1
    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time, m]
    log_results(mode, result_tests, gcd)


if __name__ == "__main__":
    main()
