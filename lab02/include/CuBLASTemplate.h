#pragma once

#include <cublas_v2.h>
#include <stdexcept>
#include <concepts>

template <typename T>
concept FloatLike = std::same_as<T, float> || std::same_as<T, double>;

template <FloatLike T>
cublasStatus_t cublasGemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                const T* alpha, const T* A, int lda, const T* x, int incx,
                const T* beta, T* y, int incy) {
    if constexpr (std::same_as<T, float>) {
        return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    } else if constexpr (std::same_as<T, double>) {
        return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <FloatLike T>
cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy) {
    if constexpr (std::same_as<T, float>) {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    } else if constexpr (std::same_as<T, double>) {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <FloatLike T>
cublasStatus_t cublasScal(cublasHandle_t handle, int n, const T* alpha, T* x, int incx) {
    if constexpr (std::same_as<T, float>) {
        return cublasSscal(handle, n, alpha, x, incx);
    } else if constexpr (std::same_as<T, double>) {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <FloatLike T>
cublasStatus_t cublasDot(cublasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result) {
    if constexpr (std::same_as<T, float>) {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    } else if constexpr (std::same_as<T, double>) {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}
