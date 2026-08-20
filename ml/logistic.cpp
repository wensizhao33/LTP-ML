#include "logistic.h"

FieldShare Logistic::w;

int Logistic::myrandom(int i) { return rand() % i; }

vector<int> Logistic::random_perm()
{
    vector<int> temp, perm;
    for (int i = 0; i < Config::config->N - Config::config->B; i++)
        temp.push_back(i);

    for (int i = 0; i < Config::config->Ep; i++)
    {                                                       // construct the random shuffled indices of every epoch
        random_shuffle(temp.begin(), temp.end(), myrandom); // random permutations of indices between 0 to N-1
        perm.insert(perm.end(), temp.begin(), temp.end());  // append the random permutated indicies
        // cout << "perm:" << perm.size() <<endl;
    }
    return perm;
}

void Logistic::next_batch(FieldShare &batch, int start, vector<int> &perm, Matrix128 &data, Matrix128 &delta_data)
{
    int col = data.cols();
    batch.eta = data.block(perm[start], 0, Config::config->B, col);
    batch.gamma = delta_data.block(perm[start], 0, Config::config->B, col);
}

void Logistic::sigmoid(Matrix128 &x)
{
    for (int i = 0; i < x.size(); i++)
    {
        if (x(i) >= 50000000000000001 && x(i) <= 99999999999967235)
            x(i) = 0;
        else if (x(i) >= 32768 && x(i) < 50000000000000001)
            x(i) = 65536;
        else
            x(i) = x(i) + 32768;
    }
}

Matrix128 Logistic::argmax(Matrix128 &x)
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

void Logistic::train_model()
{
    FieldShare x_batch(Config::config->B, Config::config->D);
    FieldShare y_batch(Config::config->B, Config::config->numClass);
    Matrix128 train_data = IOManager::train_data;
    Matrix128 train_label = IOManager::train_label;
    Matrix128 delta_train_data = IOManager::train_data_delta;
    Matrix128 delta_train_label = IOManager::train_label_delta;
    w = new FieldShare(Config::config->D, Config::config->numClass);
    w.setZero();
    cout << "weights initialized" << endl;
    vector<int> perm = random_perm();

    int start = 0;
    int iterations = 0;

    FieldShare wx(Config::config->B, Config::config->numClass);
    FieldShare wx_y(Config::config->B, Config::config->numClass);

    // 初始化offline的数据
    int m = (Config::config->K) - 1;
    lor_cmp_off1 = new Cmp_offline(Config::config->B * 2, Config::config->numClass, Config::config->K, m);
    lor_cmp_off2 = new Cmp_offline(Config::config->B * 2, Config::config->numClass, Config::config->K, 1);

    Constant::Clock *clock_train;
    clock_train = new Constant::Clock(2);

    for (int j = 0; j < Config::config->Ep; j++)
    {
        // double error = 0;
        // cout << "第" << j << "个epoch" << endl;
        for (int i = 0; i < Config::config->N / Config::config->B; i++)
        {
            next_batch(x_batch, start, perm, train_data, delta_train_data);
            next_batch(y_batch, start, perm, train_label, delta_train_label);
            start += Config::config->B;

            // Matrix128 a = Secret_Mul::Mul_reveal(x_batch);
            // Matrix128 b = Secret_Mul::Mul_reveal(w);
            // Matrix128 c = a * b;
            // Matrix128 ab = IOManager::secret_share(c);
            wx = x_batch.Mul(w);
            wx = Sigmoid(wx);
            wx_y = wx - y_batch;
            wx_y.residual();
            // 计算loss
            // MatrixXd temp = FieldShare::decode(wx_y);
            // error = error + (temp.array() * temp.array()).sum();

            FieldShare delta;

            delta = x_batch.Mul_tran(wx_y);

            w = w - delta * (double)(Config::config->R / Config::config->B);
            w.residual();
        }
        // cout << "square error" << endl;
        // cout << error / Config::config->N << endl;
        test_model();
    }
    cout << "online time:" << clock_train->get() << endl;
    cout << "it/s:" << Config::config->Ep / clock_train->get() << endl;
    // inference();
    // test_model();
}

void Logistic::test_model()
{
    double count = 0;
    Matrix128 w_(Config::config->D, Config::config->numClass);
    Matrix128 test_data = IOManager::test_data;
    Matrix128 test_label = IOManager::test_label;

    w_ = w.reveal();
    Matrix128 y_ = test_data * w_;
    y_ = y_.unaryExpr([](const u128 x)
                      { return x % P; });
    Mat::truncateMatrix128(y_);

    Logistic::sigmoid(y_);
    // Matrix128 res = argmax(y_);
    // Matrix128 label = argmax(test_label);
    for (int i = 0; i < Config::config->testN; i++)
    {
        double yyy = Constant::Util::field_to_double(y_(i, 0));
        if (yyy > 0.5 && test_label(i, 0) == Config::config->IE)
        {
            count++;
        }
        else if (yyy < 0.5 && test_label(i, 0) == 0)
        {
            count++;
        }
        // if (res(i, 0) == label(i, 0))
        // {
        //     count++;
        // }
    }
    cout << "accuracy:" << count / Config::config->testN << endl;
}

// void Logistic::inference()
// {
//     Matrix128 test_data = IOManager::test_data;
//     Matrix128 test_label = IOManager::test_label;
//     Matrix128 wx;
//     double count = 0;
//     int it = ceil(Config::config->testN / Config::config->B);
//     Matrix128 x_batch, y_batch, y_infer;

//     Matrix128 r0(Config::config->B, Config::config->D), q0(Config::config->D, Config::config->numClass), t0(Config::config->B, Config::config->numClass);
//     r0 = Secret_Mul::r0;
//     q0 = Secret_Mul::q0;
//     t0 = Secret_Mul::t0;
//     for (int i = 0; i < it; i++)
//     {
//         int temp = min(Config::config->B, Config::config->testN - i * Config::config->B);
//         x_batch = test_data.middleRows(i * Config::config->B, temp);
//         y_batch = test_label.middleRows(i * Config::config->B, temp);
//         wx = Secret_Mul::Multiply(x_batch, w, r0, q0, t0);
//         y_infer = Secret_Cmp::Sigmoid(wx);
//         Matrix128 y_predict = Secret_Mul::Mul_reveal(y_infer);
//         Matrix128 y_plaintext = Secret_Mul::Mul_reveal(y_batch);
//         Matrix128 res = argmax(y_predict);
//         Matrix128 label = argmax(y_plaintext);
//         for (int j = 0; j < temp; j++)
//         {
//             double yyy = Constant::Util::u64_to_double(y_predict(j));
//             if (yyy > 0.5)
//                 y_predict(j) = Config::config->IE;
//             else
//                 y_predict(j) = 0;
//             count = count + (y_predict(j) == y_plaintext(j));
//             // if (res(i, 0) == label(i, 0))
//             // {
//             //     count++;
//             // }
//         }
//     }
//     cout << "accuracy of inference:" << count * 1.0 / Config::config->testN << endl;
// }
