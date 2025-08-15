#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>

extern double sigma, beta, epsilon, theta, rho, V_0;

void makeGrid(double& h_s, double& h_v, double& S_max, double& V_max, int& i_, int& j_, double T,
              double M_s, double M_v, double sigma, double epsilon, double S_0, double K, int n,
              int m);

void fillMatrixA(std::vector<std::vector<double>>& A, std::vector<double>& b, int k, int n, int m,
                 double tau, double r, double d, double sigma, double beta, double epsilon,
                 double rho, double h_s, double h_v);

double payoff(double S);

inline int Q(int i, int j, int n) {
    return i * (n + 1) + j;
}

#endif  // UTILS_HPP
