#ifndef OFFLINE_H
#define OFFLINE_H

#include "../util/SocketUtil.h"
#include "../share/FieldShare.h"

extern int party;
extern u128 pow_2_i[64];
extern u128 pow_2_i_inverse[64];

// offline阶段为训练前运行，当有协助方掉线时，不用重新运行offline阶段的代码
// 因此为了测试online的效率，offline所生成的矩阵均设成常数

class Mul_offline
{
public:
    Matrix128 ab;
    Matrix128 gamma;

    static Matrix128 Reshare(Matrix128 x);
    static Matrix128 Random_matrix_for_trunc(int row, int col);
    static Matrix128 Random_matrix(int row, int col);

    static u128 pre_mul_u128(u128 &a, u128 &b, int pid1, int pid2);
    static Matrix128 cwise_triple(Matrix128 a, Matrix128 b);
    static Matrix128 cwise_triple_nparty(Matrix128 a, Matrix128 b);
    static Matrix128 pre_mul_matrix(Matrix128 &a, Matrix128 &b, int pid1, int pid2);
    static Matrix128 mul_triple(Matrix128 a, Matrix128 b);

    // for truncation
    static Matrix128 p_rand_int(int r, int c, int bit);
    static Matrix128 p_rand_bit(int r, int c);
    static void p_rand_m(int k, int m, Matrix128 &r1, Matrix128 &r2, Matrix128 **bits);
    static Matrix128 trunc_m_pr(Matrix128 eta, int k, int m);
};

class Cmp_offline
{
public:
    FieldShare r1;
    FieldShare r2;
    FieldShare **bits;

    FieldShare **r;
    FieldShare **w;
    FieldShare w_combine;
    FieldShare **r_inverse;

    Cmp_offline(int row, int col, int k, int m);
    static void presufMul(int k, int row, int col, FieldShare **w, FieldShare *w_combine, FieldShare **z);
    static FieldShare p_rand_int(int r, int c, int k);
    static FieldShare p_rand_bit(int r, int c);
    static void p_rand_m(int m, FieldShare *r2, FieldShare *r1, FieldShare **bits, int k_bit = Config::config->K);
};
#endif
