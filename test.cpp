#include <iostream>
#include "json/json.h"
#include "ml/linear.h"
#include "ml/logistic.h"
#include "ml/nn.h"
#include <omp.h>
using namespace std;
using namespace Eigen;
// using namespace emp;

u128 send_size = 0;
u128 receive_size = 0;
int rounds = 0;
double total_time = 0;

Cmp_offline *lor_cmp_off1;
Cmp_offline *lor_cmp_off2;
Cmp_offline *nn_cmp_off1;
Cmp_offline *nn_cmp_off2;
Cmp_offline *nn_cmp_off3;
Cmp_offline *nn_cmp_off4;

Matrix128 reveal(Matrix128 data)
{
    vector<Matrix128> shares(Config::config->M + Config::config->nd);
    int row = data.rows();
    int col = data.cols();
    MatrixXu data_field = data.cast<u64>();
    Matrix128 res(row, col);
    res.setZero();
    for (int i = 0; i < Config::config->M + Config::config->nd; i++)
    {
        if (i == party)
        {
            shares[i] = data;
        }
        else
        {
            SocketUtil::send(data_field, i);
        }
    }
    for (int i = 0; i < Config::config->M + Config::config->nd; i++)
    {
        if (i != party)
        {
            MatrixXu share_field;
            SocketUtil::receive(share_field, i);
            shares[i] = share_field.cast<u128>();
        }
    }
    for (int i = 0; i < Config::config->M + Config::config->nd; i++)
    {
        res += shares[i];
    }
    res = res.unaryExpr([](const u128 x)
                        { return x % P; });
    return res;
}

void test_Mul_reveal()
{
    Matrix128 temp(2, 2);
    Matrix128 gamma = Matrix128::Zero(2, 2);
    temp << 32767, 32769,
        99999999999967236, 99999999999967234;
    FieldShare share(2, 2);
    Matrix128 delta = IOManager::secret_share(temp, gamma);
    share.eta = temp;
    share.gamma = delta;
    cout << share.reveal() << endl;
}

void test_trun_pr()
{
    Matrix128 temp(3, 1);
    // temp = Mul_offline::Random_matrix_for_trunc(128, 1);
    temp << 262144, 131072, 99999999999868931;
    Matrix128 temp_share = Mul_offline::Reshare(temp);
    Matrix128 raw_temp;
    raw_temp = FieldShare::Mul_reveal(temp_share);
    cout << raw_temp << endl;
    Mat::truncateMatrix128(raw_temp);
    cout << raw_temp << endl;
    Matrix128 res;

    Constant::Clock *clock_test;
    clock_test = new Constant::Clock(1);
    res = Mul_offline::trunc_m_pr(temp_share, 20, 16);
    // cout << "online time:" << clock_test->get() << endl;

    cout << FieldShare::Mul_reveal(res) << endl;
    // Matrix128 b = Mul_offline::Reshare(res);
    // b = FieldShare::Mul_reveal(b);
    // cout << b << endl;
}

void test_mul_triple()
{
    Matrix128 a, b;
    Matrix128 c;
    Matrix128 a_raw, b_raw, c_raw;
    a = Mat::randomMatrixField(10, 4);
    b = Mat::randomMatrixField(4, 1);

    c = Mul_offline::mul_triple(a, b);

    Matrix128 x1;
    a_raw = FieldShare::Mul_reveal(a);
    b_raw = FieldShare::Mul_reveal(b);

    c_raw = FieldShare::Mul_reveal(c);
    Matrix128 ab;
    ab = (a_raw * b_raw).unaryExpr([](const u128 x)
                                   { return x % P; });
    cout << "c:" << endl
         << c_raw << endl;
    cout << "ab" << endl
         << ab << endl;
}

void test_cwise_triple()
{
    Matrix128 a, b;
    Matrix128 c;
    Matrix128 a_raw, b_raw, c_raw;
    a = Mat::randomMatrixField(1, 1);
    b = Mat::randomMatrixField(1, 1);

    if (party < Config::config->M)
    {
        c = Mul_offline::cwise_triple_nparty(a, b);
        a_raw = FieldShare::Mul_reveal_for_nparty(a);
        b_raw = FieldShare::Mul_reveal_for_nparty(b);

        c_raw = FieldShare::Mul_reveal_for_nparty(c);
        Matrix128 ab;
        ab = (a_raw.cwiseProduct(b_raw)).unaryExpr([](const u128 x)
                                                   { return x % P; });
        cout << "c:" << endl
             << c_raw << endl;
        cout << "ab" << endl
             << ab << endl;
    }
    // c = Mul_offline::cwise_triple(a, b);
    // a_raw = FieldShare::Mul_reveal(a);
    // b_raw = FieldShare::Mul_reveal(b);

    // c_raw = FieldShare::Mul_reveal(c);
    // Matrix128 ab;
    // ab = (a_raw.cwiseProduct(b_raw)).unaryExpr([](const u128 x)
    //                                            { return x % P; });
    // cout << "c:" << endl
    //      << c_raw << endl;
    // cout << "ab" << endl
    //      << ab << endl;
}

void test_reshare()
{
    Matrix128 a;
    a = Mat::randomMatrixField(1, 1);
    Matrix128 b;
    if (party == 0)
        a(0, 0) = 83299175320624528;
    else if (party == 1)
        a(0, 0) = 21503061798852351;
    else if (party == 2)
        a(0, 0) = 95199157056956057;
    else
        a(0, 0) = 94136713165376;
    // cout << "share:" << endl
    //      << b << endl;
    Matrix128 a_share = Mul_offline::Reshare(a);
    cout << "a_share: " << a_share << endl;
    cout << "reshare:" << endl
         << FieldShare::Mul_reveal(a_share) << endl;
}

void test_mul_pub()
{
    Matrix128 a, b, a_raw, aa, b_raw;
    if (party < Config::config->M)
    {
        a = Mat::randomMatrixField(2, 2);

        a_raw = FieldShare::Mul_reveal_for_nparty(a);

        cout << "a_raw:" << endl
             << a_raw << endl;

        b = FieldShare::Mulpub(a);

        b_raw = FieldShare::Mul_reveal_for_nparty(b);
        cout << "b: " << b_raw << endl;
        aa = (a_raw.cwiseProduct(a_raw)).unaryExpr([](const u128 x)
                                                   { return x % P; });
        cout << "aa: " << aa << endl;
    }
}

void test_mul()
{
    FieldShare a, b;
    a = FieldShare::Random(1, 3);
    b = FieldShare::Random(3, 3);
    // a.gamma.setConstant(0);
    // // b.gamma = Matrix128
    // b.gamma.setConstant(0);
    // a.eta << 0, 0, 100000000000000002;
    // b.eta << 100000000000000002, 0, 100000000000000002;
    // cout << a.reveal() << endl;
    // cout << b.reveal() << endl;
    if (party < Config::config->M)
    {
        a.eta = (a.eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                            { return x % P; });
        b.eta = (b.eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                            { return x % P; });
    }
    a.eta = Mul_offline::Reshare(a.eta);
    b.eta = Mul_offline::Reshare(b.eta);
    Matrix128 a_eta_raw, b_eta_raw;
    a_eta_raw = FieldShare::Mul_reveal(a.eta);
    b_eta_raw = FieldShare::Mul_reveal(b.eta);

    Matrix128 a_raw, b_raw, ab, c_raw;
    a_raw = a.reveal();
    b_raw = b.reveal();
    FieldShare c;
    c = a.Mul(b);
    c_raw = c.reveal();
    cout << "c: " << c_raw << endl;
    ab = (a_raw * b_raw).unaryExpr([](const u128 x)
                                   { return x % P; });
    Mat::truncateMatrix128(ab);
    // ab = ab / 65536;
    cout << "ab: " << ab << endl;
}

// void test_pre_mul_128()
// {
//     if (party < Config::config->M)
//     {
//         u128 a, b;
//         u128 ab;
//         a = Constant::Util::random_field();
//         b = Constant::Util::random_field();
//         ab = (Mat::coe(party, party) * a * b) % P;
//         cout << ab << endl;
//         cout << a << endl;
//         cout << b << endl;
//         // cout << FieldShare::Mul_reveal_for_nparty(a) << endl;
//         // cout << FieldShare::Mul_reveal_for_nparty(b) << endl;
//         for (int i = 0; i < Config::config->M; i++)
//         {
//             for (int j = i + 1; j < Config::config->M; j++)
//             {
//                 if ((party == i || party == j))
//                 {
//                     ab += Mul_offline::pre_mul_u128(a, b, i, j);
//                 }
//             }
//         }
//         cout << ab << endl;
//     }
// }

void test_fieldshare_offline()
{
    int m = 16;
    FieldShare r2 = FieldShare(1, 2);
    FieldShare r1 = FieldShare(1, 2);
    FieldShare **bits;
    bits = new FieldShare *[m];
    for (int i = 0; i < m; i++)
        bits[i] = new FieldShare();
    Cmp_offline::p_rand_m(m, &r2, &r1, bits);
    cout << "r1_gamma:" << r1.gamma << endl;
    cout << "r2_gamma:" << r2.gamma << endl;

    cout << "r1_raw:" << r1.reveal() << endl;
    cout << "r2_raw:" << r2.reveal() << endl;

    FieldShare temp(1, 2);
    temp.setZero();
    for (int i = 0; i < m; i++)
    {
        temp += (pow_2_i[i] * *bits[i]);
        temp.residual();
    }
    cout << "temp:" << temp.reveal() << endl;
}

void test_relu()
{
    FieldShare temp = FieldShare::Random(1, 3);
    temp.gamma.setConstant(0);
    temp.eta << 32767, 32769, 99999999999967236;
    if (party < Config::config->M)
    {
        temp.eta = (temp.eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                                  { return x % P; });
    }
    temp.eta = Mul_offline::Reshare(temp.eta);
    cout << temp.reveal() << endl;
    FieldShare res;
    res = ReLU(temp);
    Matrix128 res_raw = res.reveal();
    cout << res_raw << endl;
}

void test_sigmoid()
{
    FieldShare temp = FieldShare::Random(1, 6);
    temp.gamma.setConstant(0);
    temp.eta << 99999999999987658, 12345, 32767, 32769, 99999999999967236, 99999999999967232;
    if (party < Config::config->M)
    {
        temp.eta = (temp.eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                                  { return x % P; });
    }
    temp.eta = Mul_offline::Reshare(temp.eta);
    cout << temp.reveal() << endl;
    FieldShare res;
    res = Sigmoid(temp);
    Matrix128 res_raw = res.reveal();
    cout << res_raw << endl;
}

void test_nn_offline()
{
    // 前向mul_triple
    // 第一层(128*784, 784*64)
    Matrix128 x1(128, 784), y1(784, 64), z1(128, 64);
    z1 = Mul_offline::mul_triple(x1, y1);
    // 128*64 分成 256个小矩阵 32*1
    u128 size = send_size;
    Constant::Clock *clock_trun;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z1 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z1_ = Mul_offline::trunc_m_pr(eta_z1, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 255 * clock_trun->get();
    send_size = send_size + 255 * (send_size - size);
    cout << __LINE__ << endl;

    // 第二层(128*64, 64*64)
    Matrix128 x2(128, 64), y2(64, 64), z2(128, 64);
    z2 = Mul_offline::mul_triple(x2, y2);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z2 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z2_ = Mul_offline::trunc_m_pr(eta_z2, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 255 * clock_trun->get();
    send_size = send_size + 255 * (send_size - size);
    cout << __LINE__ << endl;

    // 第一层和第二层relu 2次comare + 2 cwise
    //  128*64 分成 256个小矩阵 32*1
    size = send_size;
    Constant::Clock *clock_com;
    clock_com = new Constant::Clock(1);
    FieldShare r2_1 = FieldShare(32, 1);
    FieldShare r1_1 = FieldShare(32, 1);
    FieldShare **bits;
    bits = new FieldShare *[19];
    for (int i = 0; i < 19; i++)
        bits[i] = new FieldShare();
    Cmp_offline::p_rand_m(19, &r2_1, &r1_1, bits);

    FieldShare **w = new FieldShare *[19];
    FieldShare **z = new FieldShare *[19];
    FieldShare w_combine(19 * 64, 1);
    for (int i = 0; i < 19; i++)
    {
        w[i] = new FieldShare();
        z[i] = new FieldShare();
    }
    Cmp_offline::presufMul(19, 32, 1, w, &w_combine, z);

    FieldShare r2_2(32, 1);
    FieldShare r1_2(32, 1);
    FieldShare **bits2 = new FieldShare *[1];
    bits2[0] = new FieldShare();
    Cmp_offline::p_rand_m(1, &r2_2, &r1_2, bits2);
    total_time = total_time + 511 * clock_com->get();
    send_size = send_size + 511 * (send_size - size);
    cout << __LINE__ << endl;

    // cwise
    Matrix128 x3(128, 64), y3(128, 64), z3(128, 64);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    z3 = Mul_offline::cwise_triple(x3, y3);
    Matrix128 eta_z3 = Mul_offline::Random_matrix(128, 64);
    total_time = total_time + clock_trun->get();
    send_size = send_size + (send_size - size);
    cout << __LINE__ << endl;

    // 输出层(128*64, 64*1)
    Matrix128 x4(128, 64), y4(64, 1), z4(128, 1);
    z4 = Mul_offline::mul_triple(x4, y4);
    // 128*1 分成 4个32*1
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z4 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z4_ = Mul_offline::trunc_m_pr(eta_z4, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 3 * clock_trun->get();
    send_size = send_size + 3 * (send_size - size);
    cout << __LINE__ << endl;

    // 反向传播
    // 输出层 两次乘法 (64*128, 128*1), (128*1,1*64)，三次trun (两次 64* 1， 一次128*64)
    Matrix128 x5(64, 128), y5(128, 1), z5(64, 1);
    z5 = Mul_offline::mul_triple(x5, y5);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z5 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z5_ = Mul_offline::trunc_m_pr(eta_z5, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 3 * clock_trun->get();
    send_size = send_size + 3 * (send_size - size);
    cout << __LINE__ << endl;

    Matrix128 x6(128, 1), y6(1, 64), z6(128, 64);
    z6 = Mul_offline::mul_triple(x6, y6);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z6 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z6_ = Mul_offline::trunc_m_pr(eta_z6, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 255 * clock_trun->get();
    send_size = send_size + 255 * (send_size - size);
    cout << __LINE__ << endl;

    // 反向两次rule 128*64
    Matrix128 x7(128, 64), y7(128, 64), z7(128, 64);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    z7 = Mul_offline::cwise_triple(x7, y7);
    Matrix128 eta_z7 = Mul_offline::Random_matrix(128, 64);
    total_time = total_time + clock_trun->get();
    send_size = send_size + (send_size - size);
    cout << __LINE__ << endl;

    // 反向linear 两次乘法 (64*128, 128*64), (128*64,64*64)，三次trun (两次 64* 64， 一次128*64)
    Matrix128 x8(64, 128), y8(128, 1), z8(64, 1);
    z8 = Mul_offline::mul_triple(x8, y8);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z8 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z8_ = Mul_offline::trunc_m_pr(eta_z8, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 255 * clock_trun->get();
    send_size = send_size + 255 * (send_size - size);

    Matrix128 x9(128, 64), y9(64, 64), z9(128, 64);
    z9 = Mul_offline::mul_triple(x9, y9);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z9 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z9_ = Mul_offline::trunc_m_pr(eta_z9, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 255 * clock_trun->get();
    send_size = send_size + 255 * (send_size - size);
    cout << __LINE__ << endl;

    // 反向输入层 1次乘法（784*128，128*64），2次trun （784*64）
    Matrix128 x10(784, 128), y10(128, 64), z10(784, 64);
    z10 = Mul_offline::mul_triple(x10, y10);
    size = send_size;
    clock_trun = new Constant::Clock(1);
    Matrix128 eta_z10 = Mul_offline::Random_matrix_for_trunc(32, 1);
    Matrix128 z10_ = Mul_offline::trunc_m_pr(eta_z10, Config::config->K, Config::config->DECIMAL_LENGTH);
    total_time = total_time + 3135 * clock_trun->get();
    send_size = send_size + 3135 * (send_size - size);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        DBGprint("Please enter party index:\n");
        scanf("%d", &party);
    }
    else
    {
        party = argv[1][0] - '0';
    }
    DBGprint("party index: %d\n", party);
    Config::config = Config::init("constant.json");
    Mat::init_public_vector();
    Constant::Util::pow_2_init();
    SocketUtil::socket_init();
    IOManager::init();

    // test_reveal();
    // test_trun_pr();
    // test_mul_pub();
    // test_reshare();
    // test_mul_triple();
    // test_cwise_triple();
    // test_mul();
    // test_relu();
    // test_sigmoid();
    // test_fieldshare_offline();
    // test_pre_mul_128();

    omp_set_num_threads(8);
    cout << __LINE__ << endl;
    Constant::Clock *clock_train;
    clock_train = new Constant::Clock(1);
    test_nn_offline();
    total_time += clock_train->get();
    cout << "time：" << total_time << endl;
    cout << "send: " << send_size << " " << "receive: " << receive_size << endl;
    return 0;
}