#ifndef MAT_H
#define MAT_H
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "../Constant.h"
#include <random>

using namespace std;
using namespace Eigen;

typedef Tensor<u128, 2, Eigen::RowMajor> Tensor128;
typedef Tensor<u128, 3> Tensor128_3;
typedef Matrix<u8, Dynamic, Dynamic> Matrix8;
typedef Matrix<u64, Dynamic, Dynamic> MatrixXu;
// linera regression
typedef Matrix<u128, Dynamic, Dynamic, Eigen::RowMajor> Matrix128;
// nn
// typedef Matrix<u128, Dynamic, Dynamic> Matrix128;
typedef Matrix<ll, Dynamic, Dynamic> Matrixint64;

class Mat
{
    // int r,c;
    // vector<u128> val;
public:
    static Tensor128 A, A_plus;
    static Matrix128 PM, PM_plus, PM_inv;
    static vector<u128> inv_for_mul;
    static vector<u128> inv_for_div;
    static vector<bool> div_flag;
    static Matrix128 coe;

    static void init_public_vector();
    static u128 determinant(Matrix128 &matrix);
    static Matrix128 adjointMatrix(Matrix128 &matrix);

    static Matrix128 randomMatrixField(int r, int c);
    static Matrix8 randomMatrix8(int r, int c);

    static Tensor128 initFromVector(vector<u128> &data, long rows, long cols);

    static void truncateMatrix128(Matrix128 &x);
    static void reshape(Tensor128 &x, long nrows, long ncols);

    static Matrix128 getFrom_pos(char *&p);
    static int toString_pos(char *p, int r, int c, vector<u128> val);

    static Matrix128 toMatrix(const Tensor128 &a);
    static Tensor128 toTensor(const Matrix128 &a);

    static Matrix128 power_matrix(const Matrix128 &a, u128 k);
    static Matrix128 inverse_matrix(const Matrix128 &a);
    // static Matrix128 op_Xor(const Matrix128 &a, const Matrix128 &b);
    // static Matrix8 op_Xor(const Matrix8 &a, const Matrix8 &b);
    // static Matrix128 op_Xor(const u128 &a, const Matrix128 &b);
    // static Matrix8 op_Xor(const u128 &a, const Matrix8 &b);
    // static Matrix128 op_And(const Matrix128 &a, const Matrix128 &b);
    // static Matrix8 op_And(const Matrix8 &a, const Matrix8 &b);
    // static Matrix128 op_And(const u128 &a, const Matrix128 &b);
    // static Matrix8 op_And(const u128 &a, const Matrix8 &b);
    // static Matrix128 op_Or(const Matrix128 &a, const Matrix128 &b);
    // static Matrix8 op_Or(const Matrix8 &a, const Matrix8 &b);
    // static Matrix128 op_Or(const u128 &a, const Matrix128 &b);
    // static Matrix8 op_Or(const u128 &a, const Matrix8 &b);
    // static Matrix128 op_shift_left(const Matrix128 &b, const int &a);
    // static Matrix8 op_shift_left(const Matrix8 &b, const int &a);
    // static Matrix128 op_shift_right(const Matrix128 &b, const int &a);
    // static Matrix8 op_shift_right(const Matrix8 &b, const int &a);
};

std::ostream &
operator<<(std::ostream &dest, Matrix128 matrix);

#endif // MAT_H
