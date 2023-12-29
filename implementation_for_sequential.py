import time
import numpy as np
from PolynomialOperations import PolynomialOperations
from common import get_polynomials, log_results


def main():
    implementation = 's'
    polynomials, polynomial_degree, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    polynomials = np.array([PolynomialOperations.polynomial_reduction_cpu(poly) for poly in polynomials])

    while len(polynomials) > 1:

        if len(polynomials) % 2 != 0:
            polynomials = np.vstack((polynomials, polynomials[-1]))

        polynomials_copy = polynomials.copy()
        results = np.array(polynomials[0:(len(polynomials)) // 2])
        results = results.copy()

        for i in range(len(polynomials) // 2):
            PolynomialOperations.process_polynomials_cpu(results, polynomials_copy, i)
        polynomials = results.copy()

    work_time = time.time() - start_time

    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time]
    log_results(mode, result_tests, polynomials[0])


if __name__ == "__main__":
    main()
