#ifndef LINEAR_H

#include "../util/Mat.h"
#include "../util/IOManager.h"
#include "../share/FieldShare.h"
#include "../core/Offline.h"

extern int party;

class Linear
{
public:
    static FieldShare w;

    static int myrandom(int i);
    static vector<int> random_perm();

    static void next_batch(FieldShare &batch, int start, vector<int> &perm, Matrix128 &data, Matrix128 &delta_data);
    static Matrix128 argmax(Matrix128 &x);
    static void train_model();
    static void test_model();
    static void inference();
};

#endif // LINEAR_H