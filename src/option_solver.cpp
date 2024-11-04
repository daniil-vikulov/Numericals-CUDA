#include "OptionSolver.hpp"
#include "utils.hpp"
#include "GaussMethod.hpp"

double solveOptionDense(int n, int m, int p, double T, double K, double S_0, double r, double d) {
    int N = (n + 1) * (m + 1);

    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0));
    std::vector<double> b(N, 0);
    std::vector<double> x(N, 0);
    GaussMethod gm(N);

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            b[Q(i, j, n)] = payoff(S_0 + j * (S_0 / n));
        }
    }

    for (int k = p; k >= 0; k--) {
        fillMatrixA(A, b, k, n, m, T / p, r, d, sigma, beta, epsilon, rho, S_0 / n, V_0 / m);
        gm.solve_slae_gauss(A, b, x);
        b = x;
    }
    return b[Q(i_, j_, n)];
}
