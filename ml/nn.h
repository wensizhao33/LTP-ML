#ifndef NN_H
#define NN_H

#include "../util/Mat.h"
#include "../util/IOManager.h"
#include "../core/FieldCmp.h"

extern int party;

extern Cmp_offline *nn_cmp_off1;
extern Cmp_offline *nn_cmp_off2;
extern Cmp_offline *nn_cmp_off3;
extern Cmp_offline *nn_cmp_off4;

class Layer
{
public:
    FieldShare input;
    FieldShare weight;
    virtual void forward(FieldShare &x) = 0;
    virtual void backward(FieldShare &delta) = 0;
    virtual void print() = 0;
};

class LinearLayer : public Layer
{
public:
    double for_time = 0, back_time = 0;
    LinearLayer(int input_size, int output_size, bool first_layer = false);
    int input_size;
    int output_size;
    bool first_layer;
    void forward(FieldShare &x);
    void backward(FieldShare &delta);
    void print()
    {
        cout << "liner_forward_time :" << for_time << endl;
        cout << "liner_backward_time :" << back_time << endl;
    }
};

class ReluLayer : public Layer
{
public:
    double for_time = 0, back_time = 0;
    FieldShare sign;
    void forward(FieldShare &x);
    void backward(FieldShare &delta);
    void print()
    {
        cout << "nn_forward_time :" << for_time << endl;
        cout << "nn_backward_time :" << back_time << endl;
    }
};

class NN
{
public:
    vector<Layer *> layers;
    vector<double> time;
    static int myrandom(int i);
    static vector<int> random_perm();
    void next_batch(FieldShare &batch, int start, vector<int> &perm, Matrix128 &data, Matrix128 &delta_data);
    Matrix128 argmax(Matrix128 &x);
    void add_layer(Layer *layer);
    void forward(FieldShare &x);
    void backward(FieldShare &delta);
    void fit(Matrix128 &x, Matrix128 &delta_x, Matrix128 &y, Matrix128 &delta_y, int epoch);
    static void train_model();
    void test_model();
    void inference(Matrix128 &x, Matrix128 &y);
};

// class NN
// {
// public:

//     static Matrix128 w, y_inf;

//     static int myrandom(int i);
//     static vector<int> random_perm();
//     static void next_batch(Matrix128 &batch, int start, vector<int> &perm, Matrix128 &data);
//     static Matrix128 argmax(Matrix128 &x);
//     static void train_model();
//     static void test_model();
//     static void inference();
// };

#endif // NN_H