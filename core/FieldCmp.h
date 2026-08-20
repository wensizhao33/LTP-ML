#ifndef FIELDCMP_H
#define FIELDCMP_H

#include "../util/Mat.h"
#include "../util/SocketUtil.h"
#include "../share/FieldShare.h"
#include "Offline.h"
#include "omp.h"

extern u128 pow_2_i[64];
extern u128 pow_2_i_inverse[64];
extern Cmp_offline *lor_cmp_off1;
extern Cmp_offline *lor_cmp_off2;
extern Cmp_offline *nn_cmp_off1;
extern Cmp_offline *nn_cmp_off2;
extern Cmp_offline *nn_cmp_off3;
extern Cmp_offline *nn_cmp_off4;

FieldShare ltz(const FieldShare &a, int k);

FieldShare div2m(const FieldShare &a, int k, int m);

FieldShare mod2m(const FieldShare &share, int m, int k_bit = Config::config->K);

FieldShare bitLT(const Matrix128 &a, FieldShare **bitShares, int k_bit);

FieldShare mod2(const FieldShare &a, int k_bit);

void sufMul(int k, FieldShare **shares, FieldShare **res);

void preMul(int k, FieldShare **shares, FieldShare **res);

// void p_rand_m(int m, FieldShare *r2, FieldShare *r1, FieldShare **bits, int k_bit = Config::config->K);

void get_premul_offline(int k, int row, int col, FieldShare **r, FieldShare **r_inverse, FieldShare **w);

// FieldShare p_rand_int(int r, int c, int k);

// FieldShare p_rand_bit(int r, int c);

FieldShare ReLU(FieldShare &share);

FieldShare Sigmoid(FieldShare &share);

#endif // FIELDCMP_H