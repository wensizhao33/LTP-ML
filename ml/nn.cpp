#include "nn.h"

int NN::myrandom(int i)
{
    return rand() % i;
}

vector<int> NN::random_perm()
{
    vector<int> temp, perm;
    for (int i = 0; i < Config::config->N - Config::config->B; i++)
        temp.push_back(i);

    for (int i = 0; i < Config::config->Ep + 1; i++)
    {                                                       // construct the random shuffled indices of every epoch
        random_shuffle(temp.begin(), temp.end(), myrandom); // random permutations of indices between 0 to N-1
        perm.insert(perm.end(), temp.begin(), temp.end());  // append the random permutated indicies
    }
    return perm;
}

void NN::next_batch(FieldShare &batch, int start, vector<int> &perm, Matrix128 &data, Matrix128 &delta_data)
{
    int col = data.cols();
    int begin = perm[start];
    batch.eta = data.block(begin, 0, Config::config->B, col);
    batch.gamma = delta_data.block(begin, 0, Config::config->B, col);
}

Matrix128 relu(Matrix128 x)
{
    for (int i = 0; i < x.size(); i++)
    {
        if (x(i) >= 0 && x(i) <= 50000000000000001)
            x(i) = x(i);
        else
            x(i) = 0;
    }
    return x;
}

Matrix128 NN::argmax(Matrix128 &x)
{
    int row = x.rows(), col = x.cols();
    Matrix128 res(row, 1);
    Matrixint64 temp = x.cast<ll>();
    for (int i = 0; i < row; i++)
    {
        int index = 0;
        ll max = temp(i, 0);
        for (int j = 1; j < col; j++)
        {
            if (temp(i, j) > max)
            {
                max = temp(i, j);
                index = j;
            }
        }

        res(i, 0) = index % 10;
    }
    return res;
}

void NN::forward(FieldShare &x)
{
    int l = this->layers.size();
    for (int i = 0; i < l; i++)
    {
        this->layers[i]->forward(x);
    }
}

void NN::backward(FieldShare &delta)
{
    int l = this->layers.size();
    for (int i = l - 1; i >= 0; i--)
    {
        this->layers[i]->backward(delta);
    }
}

void NN::fit(Matrix128 &train_data, Matrix128 &delta_train_data, Matrix128 &test_data, Matrix128 &delta_test_data, int epoch)
{
    vector<int> perm = random_perm();
    int start = 0;
    int iterations = 0;
    FieldShare x_batch(Config::config->B, Config::config->D);
    FieldShare y_batch(Config::config->B, Config::config->numClass);
    Constant::Clock *clock_train;
    clock_train = new Constant::Clock(2);
    for (int i = 0; i < epoch; i++)
    {
        // Constant::Clock *clock_train;
        // clock_train = new Constant::Clock(2);
        // double error = 0;
        for (int j = 0; j < Config::config->N / Config::config->B; j++)
        {
            next_batch(x_batch, start, perm, train_data, delta_train_data);
            next_batch(y_batch, start, perm, test_data, delta_test_data);

            start += Config::config->B;

            FieldShare output = x_batch;
            this->forward(output);
            FieldShare delta = output - y_batch;
            delta.residual();
            // MatrixXd temp = FieldShare::decode(delta);
            // error = error + (temp.array() * temp.array()).sum();
            this->backward(delta);
            // }
            // cout << "square error" << endl;
            // cout << error / Config::config->N << endl;
            // cout << "online time:" << clock_train->get() << endl;
            // for (int i = 0; i < this->layers.size(); i++)
            // {
            //     cout << "layer: " << i << endl;
            //     this->layers[i]->print();
        }
        this->test_model();
    }
    cout << "online time:" << clock_train->get() << endl;
    cout << "IT/s:" << epoch / clock_train->get() << endl;
}

void NN::add_layer(Layer *layer)
{
    this->layers.push_back(layer);
}

LinearLayer::LinearLayer(int input_size, int output_size, bool first_layer)
{
    this->input_size = input_size;
    this->output_size = output_size;
    this->first_layer = first_layer;
    this->weight = new FieldShare(input_size, output_size);
    Matrix128 w = Matrix128(input_size, output_size);
    static default_random_engine e;
    static normal_distribution<double> n(0, 0.05);
    MatrixXd m = MatrixXd::Zero(input_size, output_size).unaryExpr([](double dummy)
                                                                   { return n(e); });
    for (int i = 0; i < m.size(); i++)
        w(i) = Constant::Util::double_to_u128(m(i));
    Matrix128 gamma = Mat::randomMatrixField(input_size, output_size);
    this->weight.gamma = IOManager::secret_share(w, gamma);
    this->weight.eta = w;
}

void LinearLayer::forward(FieldShare &x)
{
    // Constant::Clock *clock_linear;
    // clock_linear = new Constant::Clock(2);
    this->input = x;
    x = x.Mul(this->weight);
    // this->for_time += clock_linear->get();
}

void LinearLayer::backward(FieldShare &delta)
{
    // Constant::Clock *clock_linear;
    // clock_linear = new Constant::Clock(2);
    FieldShare w_gradients = input.Mul_tran(delta);
    if (!first_layer)
    {
        delta = delta.Tran_mul(this->weight);
    }
    this->weight = this->weight - w_gradients * (double)(Config::config->R / Config::config->B);
    this->weight.residual();

    // this->back_time += clock_linear->get();
}

void ReluLayer::forward(FieldShare &x)
{
    // Constant::Clock *clock_relu;
    // clock_relu = new Constant::Clock(2);
    this->sign = 1 - (x < 0);

    x = x * this->sign;
    // this->for_time += clock_relu->get();
}

void ReluLayer::backward(FieldShare &delta)
{
    // Constant::Clock *clock_relu;
    // clock_relu = new Constant::Clock(2);

    delta = delta * this->sign;
    // this->back_time += clock_relu->get();
}

void NN::train_model()
{
    Matrix128 train_data = IOManager::train_data;
    Matrix128 train_label = IOManager::train_label;
    Matrix128 delta_train_data = IOManager::train_data_delta;
    Matrix128 delta_train_label = IOManager::train_label_delta;

    Matrix128 test_data = IOManager::test_data;
    Matrix128 test_label = IOManager::test_label;

    int m = (Config::config->K) - 1;
    nn_cmp_off1 = new Cmp_offline(Config::config->B, Config::config->hiddenDim, Config::config->K, m);
    nn_cmp_off2 = new Cmp_offline(Config::config->hiddenDim, Config::config->hiddenDim, Config::config->K, m);
    nn_cmp_off3 = new Cmp_offline(Config::config->B, Config::config->hiddenDim, Config::config->K, 1);
    nn_cmp_off4 = new Cmp_offline(Config::config->hiddenDim, Config::config->hiddenDim, Config::config->K, 1);

    NN net;
    Layer *linearLayer0 = new LinearLayer(Config::config->D, Config::config->hiddenDim, true);
    net.add_layer(linearLayer0);
    Layer *reluLayer0 = new ReluLayer();
    net.add_layer(reluLayer0);
    Layer *linearLayer1 = new LinearLayer(Config::config->hiddenDim, Config::config->hiddenDim);
    net.add_layer(linearLayer1);
    Layer *reluLayer1 = new ReluLayer();
    net.add_layer(reluLayer1);
    Layer *linearLayer2 = new LinearLayer(Config::config->hiddenDim, Config::config->numClass);
    net.add_layer(linearLayer2);
    // int size = net.layers.size();
    // net.time.resize(size);
    net.fit(train_data, delta_train_data, train_label, delta_train_label, Config::config->Ep);
}

void NN::test_model()
{
    double count = 0;
    Matrix128 test_data = IOManager::test_data;
    Matrix128 test_label = IOManager::test_label;

    vector<Matrix128> W_(static_cast<unsigned long>(Config::config->nLayer + 1));
    vector<Matrix128> A(static_cast<unsigned long>(Config::config->nLayer + 1));
    vector<Matrix128> Z(static_cast<unsigned long>(Config::config->nLayer + 1));
    A[0] = test_data;
    Matrix128 result;

    W_[1] = this->layers[0]->weight.reveal();
    W_[2] = this->layers[2]->weight.reveal();
    W_[3] = this->layers[4]->weight.reveal();

    for (int l = 1; l <= Config::config->nLayer; l++)
    {
        Z[l] = A[l - 1] * W_[l];
        Z[l] = Z[l].unaryExpr([](const u128 x)
                              { return x % P; });
        Mat::truncateMatrix128(Z[l]);
        if (l == Config::config->nLayer)
            A[l] = Z[l];
        else
            A[l] = relu(Z[l]);
    }
    result = A[Config::config->nLayer];
    // Matrix128 res = argmax(result);
    // Matrix128 label = argmax(test_label);
    for (int i = 0; i < Config::config->testN; i++)
    {
        double resulttt = Constant::Util::field_to_double(result(i, 0));
        if (resulttt > 0.5 && test_label(i, 0) == Config::config->IE)
            count++;
        else if (resulttt < 0.5 && test_label(i, 0) == 0)
            count++;
        // if (res(i, 0) == label(i, 0))
        // {
        //     count++;
        // }
    }
    cout << "accuracy:" << count / Config::config->testN << endl;
}

// void NN::inference(Matrix128 &x, Matrix128 &y)
// {
//     int l = y.rows();
//     int batch_len = ceil(l / Config::config->B);
//     Matrix128 x_batch, y_batch, y_batch_plaintext;
//     int count = 0;

//     for (int j = 0; j < batch_len; j++)
//     {
//         int temp = min(Config::config->B, l - j * Config::config->B);
//         x_batch = x.middleRows(j * Config::config->B, temp);
//         y_batch = y.middleRows(j * Config::config->B, temp);
//         Matrix128 predict = this->forward(x_batch);
//         predict = Secret_Mul::Mul_reveal(predict);
//         y_batch_plaintext = Secret_Mul::Mul_reveal(y_batch);
//         Matrix128 res = argmax(predict);
//         Matrix128 label = argmax(y_batch_plaintext);
//         for (int i = 0; i < temp; i++)
//         {
//             // if (predict(i) > IE / 2)
//             //     predict(i) = IE;
//             // else
//             //     predict(i) = 0;
//             // count = count + (predict(i) == y_batch_plaintext(i));
//             if (res(i, 0) == label(i, 0))
//             {
//                 count++;
//             }
//         }
//     }
//     cout << "accuracy of inference:" << count * 1.0 / l << endl;
// }