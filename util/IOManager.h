#ifndef IOMANAGER_H
#define IOMANAGER_H

#include "Mat.h"
#include "../Constant.h"
#include <cassert>
extern int party;

class IOManager
{
public:
    static Matrix128 train_data, train_label;
    static Matrix128 train_data_delta, train_label_delta;
    static Matrix128 test_data, test_label;
    static Matrix128 test_data_delta, test_label_delta;
    // static int getsize(ifstream &in);               //获取文件的列数
    static void init();
    static void load_data(ifstream &in, Matrix128 &data, Matrix128 &label, int size);
    static Matrix128 secret_share(Matrix128 &truth);
    static Matrix128 secret_share(Matrix128 &truth, Matrix128 &gamma);
    static void secret_share(Matrix128 &data, Matrix128 &label, string category);
    static void load_ss(ifstream &in, Matrix128 &data, Matrix128 &label, ifstream &in_delta, Matrix128 &data_delta, Matrix128 &label_delta, int size);
};

#endif // IOMANAGER_H