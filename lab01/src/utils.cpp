#include "utils.hpp"

double sigma = 0.2, beta = 0.5, epsilon = 0.1, theta = 0.3, rho = -0.5, V_0 = 0.04;

void makeGrid(double& h_s, double& h_v, double& S_max, double& V_max, int& i_, int& j_, double T,
              double M_s, double M_v, double sigma, double epsilon, double S_0, double K, int n,
              int m) {
    V_max = std::max(1.0, theta) * (1 + M_v * epsilon * std::sqrt(T));
    S_max = std::max(S_0, K) * std::exp(M_s * sigma * std::sqrt(T));

    h_s = S_max / static_cast<double>(n);
    h_v = V_max / static_cast<double>(m);
    i_ = static_cast<int>(std::round(S_0 / h_s));
    j_ = static_cast<int>(std::round(V_0 / h_v));
    h_s = S_0 / i_;
    h_v = V_0 / j_;
    S_max = n * h_s;
    V_max = m * h_v;
}

void fillMatrixA(std::vector<std::vector<double>>& A, std::vector<double>& b, int k, int n, int m,
                 double tau, double r, double d, double sigma, double beta, double epsilon,
                 double rho, double h_s, double h_v) {
    for (int i = 1; i <= m - 1; ++i) {
        for (int j = 1; j <= n - 1; ++j) {
            int ij = Q(i, j, n);
            double s = j * h_s;
            double v = i * h_v;

            double a_s = 0.5 * sigma * sigma * s * s / (h_s * h_s);
            double b_s = (r - d) * s / (2 * h_s);
            double a_v = 0.5 * epsilon * epsilon * v / (h_v * h_v);
            double b_v = beta * (theta - v) / (2 * h_v);
            double cross_term = rho * sigma * epsilon * s * v / (4 * h_s * h_v);
            double c = -1 / tau - (r + d);

            A[ij][ij] += -2 * (a_s + a_v) + c;
            A[ij][Q(i + 1, j, n)] += a_v + b_v;
            A[ij][Q(i - 1, j, n)] += a_v - b_v;
            A[ij][Q(i, j + 1, n)] += a_s + b_s;
            A[ij][Q(i, j - 1, n)] += a_s - b_s;
            A[ij][Q(i + 1, j + 1, n)] += cross_term;
            A[ij][Q(i - 1, j - 1, n)] += cross_term;
            A[ij][Q(i + 1, j - 1, n)] -= cross_term;
            A[ij][Q(i - 1, j + 1, n)] -= cross_term;
            b[ij] = 0;
        }
    }

    for (int j = 0; j <= n; ++j) {
        int i0 = Q(0, j, n);
        A[i0][i0] = 1;
        b[i0] = 0;
    }

    for (int j = 0; j <= n; ++j) {
        int im = Q(m, j, n);
        A[im][im] = 1;
        b[im] = 0;
    }

    for (int i = 0; i <= m; ++i) {
        int j0 = Q(i, 0, n);
        A[j0][j0] = 1;
        b[j0] = 0;
    }

    for (int i = 0; i <= m; ++i) {
        int jn = Q(i, n, n);
        A[jn][jn] = 1;
        b[jn] = std::max(0.0, i * h_v - k);
    }
}

double payoff(double S) {
    return std::max(S - K, 0.0);
}
