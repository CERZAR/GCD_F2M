import numpy as np
import time

from common import get_polynomials, log_results
from PolynomialOperations import PolynomialOperations, calculate_gcd_parallel


def main():
    implementation = 'p'
    polynomials, m, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    polynomials = PolynomialOperations.uncover_alphas(m, polynomials)
    polynomials = np.array([PolynomialOperations.polynomial_reduction_gpu(poly) for poly in polynomials])
    gcd = calculate_gcd_parallel(polynomials)

    work_time = time.time() - start_time

    polynomial_degree = len(gcd) - 1
    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time, m]
    log_results(mode, result_tests, gcd)


if __name__ == "__main__":
    main()
