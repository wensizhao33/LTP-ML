#include "Mat.h"

Tensor128 Mat::A;
Tensor128 Mat::A_plus;
Matrix128 Mat::PM;
Matrix128 Mat::PM_plus;
Matrix128 Mat::PM_inv;
vector<u128> Mat::inv_for_mul;
vector<u128> Mat::inv_for_div;
Matrix128 Mat::coe;
// vector<bool> Mat::div_flag;

u128 Mat::determinant(Matrix128 &matrix)
{
    int n = matrix.rows();
    if (n == 2)
    {
        u128 det = (matrix(0, 0) * matrix(1, 1) % P - (matrix(0, 1) * matrix(1, 0) % P) + P) % P;
        if (det < 0)
        {
            det += P;
        }
        return det;
    }
    u128 det = 0;
    u128 sign = 1;
    for (int i = 0; i < n; i++)
    {
        Matrix128 submatrix(n - 1, n - 1);
        for (int j = 1; j < n; j++)
        {
            int col = 0;
            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    submatrix(j - 1, col) = matrix(j, k);
                    col++;
                }
            }
        }
        det += ((sign * matrix(0, i)) % P * determinant(submatrix));
        det %= P;
        sign = P - sign;
    }
    if (det < 0)
    {
        det += P;
    }
    return det;
}

Matrix128 Mat::adjointMatrix(Matrix128 &matrix)
{
    int n = matrix.rows();
    Matrix128 adj(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Matrix128 submatrix(n - 1, n - 1);
            int subi = 0, subj = 0;
            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    subj = 0;
                    for (int l = 0; l < n; l++)
                    {
                        if (l != j)
                        {
                            submatrix(subi, subj) = matrix(k, l);
                            subj++;
                        }
                    }
                    subi++;
                }
            }
            u128 sign = ((i + j) % 2 == 0) ? 1 : P - 1;
            adj(j, i) = sign * Mat::determinant(submatrix) % P;
        }
    }
    return adj;
}

void Mat::init_public_vector()
{
    A = Tensor128(Config::config->M, Config::config->M);
    A_plus = Tensor128(Config::config->M + Config::config->nd, Config::config->M);
    PM = Matrix128(Config::config->M, Config::config->M);
    PM_plus = Matrix128(Config::config->M + Config::config->nd, Config::config->M);
    PM_inv = Matrix128(Config::config->M, Config::config->M);
    inv_for_mul.resize(Config::config->M);
    inv_for_div.resize(Config::config->M);
    coe = Matrix128(Config::config->M, Config::config->M);
    // div_flag.resize(Config::config->M);
    // A_plus_temp << 1, 1, Uu128_MAX,
    //     1, 2, Uu128_MAX,
    //     1, 1, 0,
    //     2, 3, Uu128_MAX;
    // A_plus.setValues({{1, 0, 1}, {2, 2, Uu128_MAX - 2}, {3, 3, Uu128_MAX - 3}, {1, 1, Uu128_MAX}});
    // MatrixXi public_matrix(Config::config->M + Config::config->nd, Config::config->M);
    for (int i = 0; i < Config::config->M + Config::config->nd; i++)
    {
        for (int j = 0; j < Config::config->M; j++)
            PM_plus(i, j) = (u128)pow(i + 1, j);
    }
    cout << "public_matrix:" << PM_plus;
    PM = PM_plus.topRows(Config::config->M);
    A_plus = Mat::toTensor(PM_plus);
    u128 det = determinant(PM);
    // cout << "the determinant of public matrix is:" << (det % P) << endl;
    u128 det_inv = Constant::Util::inverse(det);
    Matrix128 adj = adjointMatrix(PM);
    for (int i = 0; i < Config::config->M; i++)
    {
        for (int j = 0; j < Config::config->M; j++)
        {
            PM_inv(i, j) = (adj(i, j) * det_inv) % P;
        }
    }
    cout << "multiply coefficent:" << endl;
    for (int i = 0; i < Config::config->M; i++)
    {
        inv_for_mul[i] = PM_inv(0, i);
        cout << inv_for_mul[i] << " ";
        inv_for_div[i] = Constant::Util::inverse(PM_inv(0, i));
    }
    cout << endl;
    cout << "division coefficent:" << endl;
    for (int i = 0; i < Config::config->M; i++)
    {
        cout << inv_for_div[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < Config::config->M; i++)
    {
        for (int j = i; j < Config::config->M; j++)
        {
            coe(i, j) = inv_for_mul[i] * inv_for_mul[j] % P;
        }
    }
}

Matrix128 Mat::randomMatrixField(int rows, int cols)
{
    Matrix128 res(rows, cols);
    res.setZero();
    u128 *data = res.data();
    for (int i = 0; i < res.size(); i++)
    {
        *(data + i) = Constant::Util::random_field();
    }
    return res;
}

Matrix8 Mat::randomMatrix8(int rows, int cols)
{
    Matrix8 res(rows, cols);
    res.setZero();
    u8 *data = res.data();
    for (int i = 0; i < res.size(); i++)
    {
        *(data + i) = rand() % 512;
    }
    return res;
}

void Mat::truncateMatrix128(Matrix128 &x)
{
    for (long i = 0; i < x.size(); i++)
    {
        if (x.data()[i] > P / 2)
            x.data()[i] = P - static_cast<u128>((P - x.data()[i]) / Config::config->IE);
        else
            x.data()[i] = static_cast<u128>(x.data()[i] / Config::config->IE);
    }
}

void Mat::reshape(Tensor128 &x, long nrows, long ncols)
{
    int row = x.dimension(0);
    int col = x.dimension(1);
    if (row == nrows && col == ncols)
    {
        return;
    }
    else if (row > nrows && col > ncols)
    {
        Matrix128 temp = toMatrix(x);
        temp = temp.topLeftCorner(nrows, ncols);
        x = toTensor(temp);
    }
    else
    {
        Matrix128 nx = Matrix128::Zero(nrows, ncols);
        Matrix128 temp = toMatrix(x);
        nx.topLeftCorner(row, col) = temp;
        Tensor128 temp1 = toTensor(nx);
        x = temp1;
    }
}

Matrix128 Mat::toMatrix(const Tensor128 &a)
{
    int row = a.dimension(0);
    int col = a.dimension(1);
    Matrix128 res(row, col);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    return res;
}

Tensor128 Mat::toTensor(const Matrix128 &a)
{
    int row = a.rows();
    int col = a.cols();
    Tensor128 res(row, col);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    return res;
}

Matrix128 Mat::power_matrix(const Matrix128 &a, u128 k)
{
    int r = a.rows();
    int c = a.cols();
    Matrix128 res(r, c);
    res.setOnes();
    if (k == 0)
    {
        return res;
    }
    Matrix128 temp = a;
    while (k > 0)
    {
        if (k & 1)
        {
            res = res.cwiseProduct(temp);
            res = res.unaryExpr([](const u128 x)
                                { return x % P; });
        }
        temp = temp.cwiseProduct(temp);
        temp = temp.unaryExpr([](const u128 x)
                              { return x % P; });
        k = k >> 1;
    }
    return res;
}

Matrix128 Mat::inverse_matrix(const Matrix128 &a)
{
    return power_matrix(a, P - 2);
}

// Matrix128 Mat::op_Xor(const Matrix128 &a, const Matrix128 &b)
// {
//     if (a.rows() == b.rows() && a.cols() == b.cols())
//     {
//         int row = a.rows();
//         int col = a.cols();
//         Matrix128 c(row, col);
//         for (int i = 0; i < a.size(); i++)
//         {
//             c(i) = a(i) ^ b(i);
//         }
//         return c;
//     }
//     else
//     {
//         cout << "The Matrices can not do operator xor" << endl;
//     }
// }

// Matrix8 Mat::op_Xor(const Matrix8 &a, const Matrix8 &b)
// {
//     if (a.rows() == b.rows() && a.cols() == b.cols())
//     {
//         int row = a.rows();
//         int col = a.cols();
//         Matrix8 c(row, col);
//         for (int i = 0; i < a.size(); i++)
//         {
//             c(i) = a(i) ^ b(i);
//         }
//         return c;
//     }
//     else
//     {
//         cout << "The Matrices can not do operator xor" << endl;
//     }
// }

// Matrix128 Mat::op_Xor(const u128 &a, const Matrix128 &b)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix128 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = a ^ b(i);
//     }
//     return c;
// }

// Matrix8 Mat::op_Xor(const u128 &a, const Matrix8 &b)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix8 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = a ^ b(i);
//     }
//     return c;
// }

// Matrix128 Mat::op_And(const Matrix128 &a, const Matrix128 &b)
// {
//     if (a.rows() == b.rows() && a.cols() == b.cols())
//     {
//         int row = a.rows();
//         int col = a.cols();
//         Matrix128 c(row, col);
//         for (int i = 0; i < a.size(); i++)
//         {
//             c(i) = a(i) & b(i);
//         }
//         return c;
//     }
//     else
//     {
//         cout << "The Matrices can not do operator and" << endl;
//     }
// }

// Matrix8 Mat::op_And(const Matrix8 &a, const Matrix8 &b)
// {
//     if (a.rows() == b.rows() && a.cols() == b.cols())
//     {
//         int row = a.rows();
//         int col = a.cols();
//         Matrix8 c(row, col);
//         for (int i = 0; i < a.size(); i++)
//         {
//             c(i) = a(i) & b(i);
//         }
//         return c;
//     }
//     else
//     {
//         cout << "The Matrices can not do operator and" << endl;
//     }
// }

// Matrix128 Mat::op_And(const u128 &a, const Matrix128 &b)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix128 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = a & b(i);
//     }
//     return c;
// }

// Matrix8 Mat::op_And(const u128 &a, const Matrix8 &b)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix8 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = a & b(i);
//     }
//     return c;
// }

// Matrix128 Mat::op_Or(const Matrix128 &a, const Matrix128 &b)
// {
//     if (a.rows() == b.rows() && a.cols() == b.cols())
//     {
//         int row = a.rows();
//         int col = a.cols();
//         Matrix128 c(row, col);
//         for (int i = 0; i < a.size(); i++)
//         {
//             c(i) = a(i) | b(i);
//         }
//         return c;
//     }
//     else
//     {
//         cout << "The Matrices can not do operator and" << endl;
//     }
// }

// Matrix8 Mat::op_Or(const Matrix8 &a, const Matrix8 &b)
// {
//     if (a.rows() == b.rows() && a.cols() == b.cols())
//     {
//         int row = a.rows();
//         int col = a.cols();
//         Matrix8 c(row, col);
//         for (int i = 0; i < a.size(); i++)
//         {
//             c(i) = a(i) | b(i);
//         }
//         return c;
//     }
//     else
//     {
//         cout << "The Matrices can not do operator and" << endl;
//     }
// }

// Matrix128 Mat::op_Or(const u128 &a, const Matrix128 &b)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix128 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = a | b(i);
//     }
//     return c;
// }

// Matrix8 Mat::op_Or(const u128 &a, const Matrix8 &b)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix8 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = a | b(i);
//     }
//     return c;
// }

// Matrix128 Mat::op_shift_left(const Matrix128 &b, const int &a)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix128 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = b(i) << a;
//     }
//     return c;
// }

// Matrix8 Mat::op_shift_left(const Matrix8 &b, const int &a)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix8 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = b(i) << a;
//     }
//     return c;
// }

// Matrix128 Mat::op_shift_right(const Matrix128 &b, const int &a)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix128 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = b(i) >> a;
//     }
//     return c;
// }

// Matrix8 Mat::op_shift_right(const Matrix8 &b, const int &a)
// {
//     int row = b.rows();
//     int col = b.cols();
//     Matrix8 c(row, col);
//     for (int i = 0; i < b.size(); i++)
//     {
//         c(i) = b(i) >> a;
//     }
//     return c;
// }

std::ostream &
operator<<(std::ostream &dest, Matrix128 matrix)
{
    std::ostream::sentry s(dest);
    int row = matrix.rows();
    int col = matrix.cols();
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            dest << matrix(i, j);
            dest << " ";
        }
        dest << endl;
    }
    return dest;
}