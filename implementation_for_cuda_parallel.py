from numba import cuda
import numpy as np
import time
from common import get_polynomials, log_results
from PolynomialOperations import PolynomialOperations


def main():
    implementation = 'p'
    polynomials, polynomial_degree, polynomials_amount, mode = get_polynomials(implementation)

    start_time = time.time()

    polynomials = np.array([PolynomialOperations.polynomial_reduction_gpu(poly) for poly in polynomials])

    while len(polynomials) > 1:

        if len(polynomials) % 2 != 0:
            polynomials = np.vstack((polynomials, polynomials[-1]))

        cudapolynomials = cuda.to_device(polynomials)
        results = np.array(polynomials[0:(len(polynomials)) // 2])
        results = cuda.to_device(results)
        PolynomialOperations.process_polynomials_gpu.forall(len(polynomials) // 2)(results, cudapolynomials)
        polynomials = results.copy_to_host()

    work_time = time.time() - start_time

    result_tests = [implementation, polynomial_degree, polynomials_amount, work_time]
    log_results(mode, result_tests, polynomials[0])


if __name__ == "__main__":
    main()
