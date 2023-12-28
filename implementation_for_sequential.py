import time
import numpy as np
from PolynomialOperations import PolynomialOperations
from common import get_polynomials, log_results, reduction_polynomials


def process_polynomials(res, polynomials_copy, polynomial_id):
    if polynomial_id < polynomials_copy.shape[0]:
        PolynomialOperations.calculate_polynomial_gcd(
            polynomials_copy[2 * polynomial_id],
            polynomials_copy[2 * polynomial_id + 1],
            res,
            polynomial_id
        )


def main():
    implementation = 's'
    polynomials, polynomial_degree, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    polynomials = [reduction_polynomials(poly) for poly in polynomials]
    polynomials = np.array(polynomials)

    while len(polynomials) > 1:

        if len(polynomials) % 2 != 0:
            polynomials = np.vstack((polynomials, polynomials[-1]))

        polynomials_copy = polynomials.copy()
        results = np.array(polynomials[0:(len(polynomials)) // 2])
        results = results.copy()

        for i in range(len(polynomials) // 2):
            process_polynomials(results, polynomials_copy, i)
        polynomials = results.copy()

    work_time = time.time() - start_time

    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time]
    log_results(mode, result_tests, polynomials[0])


if __name__ == "__main__":
    main()
