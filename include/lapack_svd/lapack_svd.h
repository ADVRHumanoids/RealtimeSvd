#ifndef __LAPACK_EIGEN_SVD_H__
#define __LAPACK_EIGEN_SVD_H__

#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <lapacke.h>
#include <vector>


class LapackSvd
{
    
public:
    
    LapackSvd();
    LapackSvd(int n, int m);
    
    template <typename Derived>
    bool compute(const Eigen::MatrixBase<Derived>& A);
    
    const Eigen::MatrixXd& matrixU() const;
    const Eigen::MatrixXd& matrixV() const;
    const Eigen::VectorXd& singularValues() const;
    
private:
    
    void allocate_workspace();
    
    int _rows;
    int _cols;
    
    std::vector<int> _i_work;
    std::vector<double> _d_work;
    
    Eigen::VectorXd _sv;
    Eigen::MatrixXd _U, _V;
    Eigen::MatrixXd _A;
};



/* Implementation */

inline LapackSvd::LapackSvd():
    _rows(-1),
    _cols(-1)
{

}

inline LapackSvd::LapackSvd(int n, int m):
    _rows(n),
    _cols(m)
{
    allocate_workspace();
}


inline void LapackSvd::allocate_workspace()
{
    if( !(_rows > 0 && _cols > 0) )
    {
        throw std::runtime_error("lapack svd: invalid size");
    }
    
    _sv.setZero(std::min(_rows, _cols));
    _U.setZero(_rows, _rows);
    _V.setZero(_cols, _cols);
    _A.setZero(_rows, _cols);
    
    
    double work_size = -1;
    int lwork = -1;
    _i_work.resize(8 * _sv.size());
    int info = 0;
    
    double *s = _sv.data(), *u = _U.data(), *v = _V.data();
    int n = _rows, m = _cols;
    
    char job[2] = {'A', '\0'};

    // Call dgesdd_ with lwork = -1 to query optimal workspace size:
    dgesdd_(job,            // #1   Job (A -> compute complete SVD)
            &n,             // #2   Rows
            &m,             // #3   Cols
            _A.data(),      // #4   Pointer to matrix data
            &n,             // #5   Matrix leading dimension (A column major -> put column size)
            s,              // #6   Buffer for singular values vector (size: min(n,m))
            u,              // #7   Buffer for matrix U (size: n x n)
            &n,             // #8   Leading dimension for U
            v,              // #9   Buffer for matrix Vt (size: m x m)
            &m,             // #10  Leading dimension for Vt
            &work_size,     // #11  Workspace (if lwork = 1, it returns optimal workspace size)
            &lwork,         // #12  Size of workspace (if lwork = 1, arg #11 returns optimal workspace size)
            _i_work.data(), // #13  Workspace
            &info);         // #14  Status information
    
    if (info)
    {
        std::runtime_error("error - invalid argument was passed to dgesdd_");
    }

    // Optimal workspace size is returned in work[0].
    lwork = work_size * 2;
    _d_work.resize(lwork);
}

template <typename Derived>
inline bool LapackSvd::compute(const Eigen::MatrixBase<Derived>& A)
{
    if( A.rows() != _rows || A.cols() != _cols )
    {
        throw std::invalid_argument("A.rows() != _rows || A.cols() != _cols");
    }
    
    _A.noalias() = A;
    
    double *s = _sv.data(), *u = _U.data(), *v = _V.data();
    double *work = _d_work.data();
    int *iwork = _i_work.data();
    int work_size = _d_work.size();
    int info = 0;
    int n = _rows, m = _cols;
    char job[2] = {'A', '\0'};
    
    dgesdd_(job,             // #1   Job (A -> compute complete SVD)
            &n,              // #2   Rows
            &m,              // #3   Cols
            _A.data(),       // #4   Pointer to matrix data
            &n,              // #5   Matrix leading dimension (A column major -> put column size)
            s,               // #6   Buffer for singular values vector (size: min(n,m))
            u,               // #7   Buffer for matrix U (size: n x n)
            &n,              // #8   Leading dimension for U
            v,               // #9   Buffer for matrix Vt (size: m x m)
            &m,              // #10  Leading dimension for Vt
            work,            // #11  Workspace (if lwork = 1, it returns optimal workspace size)
            &work_size,      // #12  Size of workspace (if lwork = 1, arg #11 returns optimal workspace size)
            _i_work.data(),  // #13  Workspace
            &info);          // #14  Status information
    
    _V.transposeInPlace();
    
    if(info)
    {
        return false;
    }
    else
    {
        return true;
    }
}

inline const Eigen::MatrixXd& LapackSvd::matrixU() const
{
    return _U;
}

inline const Eigen::MatrixXd& LapackSvd::matrixV() const
{
    return _V;
}

inline const Eigen::VectorXd& LapackSvd::singularValues() const
{
    return _sv;
}


#endif