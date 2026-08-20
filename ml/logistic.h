#ifndef LOGISTIC_H
#define LOGISTIC_H

#include "../util/Mat.h"
#include "../util/IOManager.h"
#include "../core/FieldCmp.h"

extern int party;
extern Cmp_offline *lor_cmp_off1;
extern Cmp_offline *lor_cmp_off2;

class Logistic
{
public:
    static FieldShare w;

    static int myrandom(int i);
    static vector<int> random_perm();

    static void sigmoid(Matrix128 &x);

    static Matrix128 argmax(Matrix128 &x);

    static void next_batch(FieldShare &batch, int start, vector<int> &perm, Matrix128 &data, Matrix128 &delta_data);
    static void train_model();
    static void test_model();
    static void inference();
};

#endif // LOGISTIC_H