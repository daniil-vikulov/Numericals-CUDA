#ifndef VECTOR_UTILS_HPP
#define VECTOR_UTILS_HPP

#include <vector>
#include <iostream>
#include <string>

enum VectorOrientation { COLUMN, ROW };

// Print a raw array (as a column or row)
void print_array(const double *a, int size, VectorOrientation o = VectorOrientation::COLUMN,
                 std::string end = "\n");

// Print a 1D vector using std::vector
void print_vector(const std::vector<double> &v, VectorOrientation o = VectorOrientation::COLUMN,
                  std::string end = "\n");

// Print a 2D matrix using a 1D array with LAPACK-compatible layout (column-major order)
void print_matrix(const double *m, int rows, int cols, std::string row_sep = ",\n",
                  std::string end = "\n");

// Print a matrix using std::vector of std::vectors
void print_matrix(const std::vector<std::vector<double>> &m, std::string row_sep = ",\n",
                  std::string end = "\n");

// Convert a std::vector of std::vectors to a raw array in LAPACK-compatible format
void vv_to_array(const std::vector<std::vector<double>> &vecvec, double *buf);

#endif  // VECTOR_UTILS_HPP
