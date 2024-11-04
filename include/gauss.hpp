#ifndef GAUSS_METHOD_HPP
#define GAUSS_METHOD_HPP

#include <vector>
#include <algorithm>
#include <iostream>

class gauss {
   private:
    int N;

   public:
    gauss(int N_) : N(N_) {}
    static void solve_slae_gauss(std::vector<std::vector<double>> A, std::vector<double> b,
                                 std::vector<double> &x);
    static void solve_slae_lapack(std::vector<std::vector<double>> &A, std::vector<double> &b,
                                  std::vector<double> &x);
};

#endif  // GAUSS_METHOD_HPP
