#include "FieldShare.h"
// #include "../core/Offline.h"
#include "../util/SocketUtil.h"
#include "../core/FieldCmp.h"

FieldShare::FieldShare()
{
}

FieldShare::FieldShare(int r, int c)
{
    this->eta = Matrix128::Zero(r, c);
    this->gamma = Matrix128::Zero(r, c);
}

FieldShare::FieldShare(const Matrix128 &eta, const Matrix128 &gamma)
{
    this->eta = eta;
    this->gamma = gamma;
}

int FieldShare::size() const
{
    return this->eta.size();
}

int FieldShare::rows() const
{
    return this->eta.rows();
}

int FieldShare::cols() const
{
    return this->eta.cols();
}

void FieldShare::resize(int r, int c)
{
    this->eta.resize(r, c);
    this->gamma.resize(r, c);
}

void FieldShare::setZero()
{
    this->eta.setZero();
    this->gamma.setZero();
}

void FieldShare::setConstant(int c)
{
    this->eta.setZero();
    this->gamma.setConstant(c);
}

// FieldShare FieldShare::transpose()
// {
//     Matrix128 temp0, temp1;
//     temp0 = this->eta.transpose();
//     temp1 = this->gamma.transpose();
//     FieldShare res(temp0, temp1);
//     return res;
// }

FieldShare FieldShare::block1(int startRow, int startCol, int blockRows, int blockCols)
{
    FieldShare res(blockRows, blockCols);
    res.eta = eta.block(startRow, startCol, blockRows, blockCols);
    res.gamma = gamma.block(startRow, startCol, blockRows, blockCols);
    return res;
}

void FieldShare::block2(int startRow, int startCol, int blockRows, int blockCols, FieldShare &share)
{
    eta.block(startRow, startCol, blockRows, blockCols) = share.eta;
    gamma.block(startRow, startCol, blockRows, blockCols) = share.gamma;
}

MatrixXd FieldShare::decode(FieldShare &share)
{
    Matrix128 tmp = share.reveal();
    Matrixint64 tmp2;
    tmp2 = tmp.unaryExpr([](const u128 x)
                         { if (x > P/2) return (ll)(x - P); else return (ll)x; });
    MatrixXd res;
    res = tmp2.unaryExpr([](const ll x)
                         { return (double)(x * 1.0) / Config::config->IE; });
    return res;
}

void FieldShare::residual()
{
    this->eta = this->eta.unaryExpr([](const u128 x)
                                    { return x % P; });
    this->gamma = this->gamma.unaryExpr([](const u128 x)
                                        { return x % P; });
}

FieldShare FieldShare::Random(int row, int col)
{
    FieldShare res(row, col);
    res.gamma.setConstant(0);
    res.eta = Mat::randomMatrixField(row, col);
    return res;
}

// 用于offline 生成p_rand_bit
Matrix128 FieldShare::Mulpub_for_nparty(const FieldShare &fieldshare)
{
    int row = fieldshare.rows();
    int col = fieldshare.cols();

    Matrix128 eta_xy = Mul_offline::cwise_triple_nparty(eta, fieldshare.eta);

    Matrix128 gamma_x_y = gamma.cwiseProduct(fieldshare.eta);
    Matrix128 x_gamma_y = eta.cwiseProduct(fieldshare.gamma);
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    gamma_z_share += eta_xy;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal_for_nparty(gamma_z_share);
    gamma_z = gamma_z + gamma.cwiseProduct(fieldshare.gamma);
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });

    return gamma_z;
}

FieldShare FieldShare::Cwise_for_nparty(const FieldShare &fieldshare)
{
    int row = fieldshare.rows();
    int col = fieldshare.cols();

#ifdef TEST_FOR_TOTAL_TIME
    Matrix128 eta_z = Mat::randomMatrixField(row, col);
#else
    Matrix128 eta_z = Matrix128::Zero(row, col);
#endif
    Matrix128 eta_xy = Mul_offline::cwise_triple_nparty(eta, fieldshare.eta);

    Matrix128 gamma_x_y = gamma.cwiseProduct(fieldshare.eta);
    Matrix128 x_gamma_y = eta.cwiseProduct(fieldshare.gamma);
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    gamma_z_share += eta_xy + eta_z;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal_for_nparty(gamma_z_share);
    gamma_z = gamma_z + gamma.cwiseProduct(fieldshare.gamma);
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });

    FieldShare res(eta_z, gamma_z);
    return res;
}

Matrix128 FieldShare::Mulpub(const FieldShare &fieldshare)
{
    int row = fieldshare.rows();
    int col = fieldshare.cols();

    Matrix128 eta_xy = Mul_offline::cwise_triple(eta, fieldshare.eta);

    Matrix128 gamma_x_y = gamma.cwiseProduct(fieldshare.eta);
    Matrix128 x_gamma_y = eta.cwiseProduct(fieldshare.gamma);
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    gamma_z_share += eta_xy;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal(gamma_z_share);
    gamma_z = gamma_z + gamma.cwiseProduct(fieldshare.gamma);
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });

    return gamma_z;
}

// 两个相同的数相乘
Matrix128 FieldShare::Mulpub(const Matrix128 &x)
{
    int row = x.rows();
    int col = x.cols();
#ifdef TEST_FOR_TOTAL_TIME
    Matrix128 r = Mat::randomMatrixField(row, col);
#else
    // 测试online时长，所有的随机数置零处理
    Matrix128 r = Matrix128::Zero(row, col);
    // Matrix128 t = Matrix128::Zero(row, col);
#endif
    Matrix128 t = Mul_offline::cwise_triple_nparty(r, r);
    Matrix128 result;
    Matrix128 x_plus_r = x + r;
    x_plus_r = x_plus_r.unaryExpr([](const u128 x)
                                  { return x % P; });
    Matrix128 X_plus_R;
    X_plus_R = FieldShare::Mul_reveal_for_nparty(x_plus_r);
    result = (2 * r.cwiseProduct(X_plus_R)).unaryExpr([](const u128 x)
                                                      { return P - (x % P); });
    result = result.array() + X_plus_R.cwiseProduct(X_plus_R).array() + t.array();
    result = result.unaryExpr([](const u128 x)
                              { return x % P; });
    return result;
}

// 所有参与方将share值发送给P0，P0恢复完发给其他参与方
// Matrix128 FieldShare::Mul_reveal(const Matrix128 &data)
// {
//     int row = data.rows();
//     int col = data.cols();
//     MatrixXu data_field = data.cast<u64>();
//     Matrix128 res(row, col);
//     res.setZero();
//     if (party == 0)
//     {
//         MatrixXu share_field;
//         vector<Matrix128> shares(Config::config->M);
//         // shares[0] = data;
//         res = data * Mat::inv_for_mul[0];
//         for (int i = 1; i < Config::config->M; i++)
//         {
//             SocketUtil::receive(share_field, i);
//             shares[i] = share_field.cast<u128>();
//             res += shares[i] * Mat::inv_for_mul[i];
//         }
//         res = res.unaryExpr([](const u128 x)
//                             { return x % P; });

//         MatrixXu temp = res.cast<u64>();
//         for (int i = 1; i < Config::config->M + Config::config->nd; i++)
//         {
//             SocketUtil::send(temp, i);
//         }
//     }
//     else
//     {
//         MatrixXu temp;
//         if (party < Config::config->M)
//         {
//             SocketUtil::send(data_field, 0);
//             SocketUtil::receive(temp, 0);
//             res = temp.cast<u128>();
//         }
//         else
//         {
//             SocketUtil::receive(temp, 0);
//             res = temp.cast<u128>();
//         }
//     }
//     return res;
// }

// 所有参与方都发送share给其他参与方，通信量较大，但是通信轮数少，采用
Matrix128 FieldShare::Mul_reveal(const Matrix128 &data)
{
    vector<Matrix128> shares(Config::config->M);
    int row = data.rows();
    int col = data.cols();
    MatrixXu data_field = data.cast<u64>();
    Matrix128 res(row, col);
    res.setZero();

    if (party < Config::config->M)
    {
        for (int i = 0; i < Config::config->M; i++)
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
        for (int i = 0; i < Config::config->M; i++)
        {
            if (i != party)
            {
                MatrixXu share_field;
                SocketUtil::receive(share_field, i);
                shares[i] = share_field.cast<u128>();
            }
        }
        for (int i = 0; i < Config::config->M; i++)
        {
            res += shares[i] * Mat::inv_for_mul[i];
        }
        res = res.unaryExpr([](const u128 x)
                            { return x % P; });
    }
    if (party == 0)
    {
        MatrixXu temp = res.cast<u64>();
        for (int i = Config::config->M; i < Config::config->M + Config::config->nd; i++)
        {
            SocketUtil::send(temp, i);
        }
    }
    if (party >= Config::config->M)
    {
        MatrixXu temp;
        temp.setZero();
        SocketUtil::receive(temp, 0);
        res = temp.cast<u128>();
    }
    return res;
}

Matrix128 FieldShare::Mul_reveal_for_nparty(const Matrix128 &data)
{
    vector<Matrix128> shares(Config::config->M);
    int row = data.rows();
    int col = data.cols();
    MatrixXu data_field = data.cast<u64>();
    Matrix128 res(row, col);
    res.setZero();

    for (int i = 0; i < Config::config->M; i++)
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
    for (int i = 0; i < Config::config->M; i++)
    {
        if (i != party)
        {
            MatrixXu share_field;
            SocketUtil::receive(share_field, i);
            shares[i] = share_field.cast<u128>();
        }
    }
    for (int i = 0; i < Config::config->M; i++)
    {
        res += shares[i] * Mat::inv_for_mul[i];
    }
    res = res.unaryExpr([](const u128 x)
                        { return x % P; });
    return res;
}

FieldShare FieldShare::truncation()
{
    int row = this->eta.rows();
    int col = this->eta.cols();
#ifdef TEST_FOR_TOTAL_TIME
    Matrix128 z_ = Mul_offline::trunc_m_pr(eta, Config::config->K, Config::config->DECIMAL_LENGTH);
#else
    Matrix128 z_ = Matrix128::Zero(row, col);
#endif
    Mat::truncateMatrix128(this->gamma);
    FieldShare res(z_, this->gamma);
    return res;
}

// 所有参与方将share值发送给P0，P0恢复完发给其他参与方
// Matrix128 FieldShare::reveal()
// {
//     int row = eta.rows();
//     int col = eta.cols();
//     MatrixXu data_field = eta.cast<u64>();
//     Matrix128 res = Matrix128::Zero(row, col);
//     if (party == 0)
//     {
//         MatrixXu share_field;
//         vector<Matrix128> shares(Config::config->M);
//         // shares[0] = eta;
//         res = eta * Mat::inv_for_mul[0];
//         for (int i = 1; i < Config::config->M; i++)
//         {
//             SocketUtil::receive(share_field, i);
//             shares[i] = share_field.cast<u128>();
//             res += shares[i] * Mat::inv_for_mul[i];
//         }
//         res = res.unaryExpr([](const u128 x)
//                             { return x % P; });
//         res = P + gamma.array() - res.array();
//         res = res.unaryExpr([](const u128 x)
//                             { return x % P; });
//         MatrixXu temp = res.cast<u64>();
//         for (int i = 1; i < Config::config->M + Config::config->nd; i++)
//         {
//             SocketUtil::send(temp, i);
//         }
//     }
//     else
//     {
//         MatrixXu temp;
//         if (party < Config::config->M)
//         {
//             SocketUtil::send(data_field, 0);
//             SocketUtil::receive(temp, 0);
//             res = temp.cast<u128>();
//         }
//         else
//         {
//             SocketUtil::receive(temp, 0);
//             res = temp.cast<u128>();
//         }
//     }
//     return res;
// }

// 所有参与方都发送share给其他参与方，通信量较大，但是通信轮次少，采用
Matrix128 FieldShare::reveal()
{
    int row = eta.rows();
    int col = eta.cols();
    vector<Matrix128> shares(Config::config->M);
    MatrixXu data_field = eta.cast<u64>();
    Matrix128 res = Matrix128::Zero(row, col);
    // res.setZero();
    if (party < Config::config->M)
    {
        for (int i = 0; i < Config::config->M; i++)
        {
            if (i == party)
            {
                shares[i] = eta;
            }
            else
            {
                SocketUtil::send(data_field, i);
            }
        }
        for (int i = 0; i < Config::config->M; i++)
        {
            if (i != party)
            {
                MatrixXu share_field;
                SocketUtil::receive(share_field, i);
                shares[i] = share_field.cast<u128>();
            }
        }
        for (int i = 0; i < Config::config->M; i++)
        {
            res += shares[i] * Mat::inv_for_mul[i];
        }
        res = res.unaryExpr([](const u128 x)
                            { return x % P; });
        res = P + gamma.array() - res.array();
        res = res.unaryExpr([](const u128 x)
                            { return x % P; });
    }
    if (party == 0)
    {
        MatrixXu temp = res.cast<u64>();
        for (int i = Config::config->M; i < Config::config->M + Config::config->nd; i++)
        {
            SocketUtil::send(temp, i);
        }
    }
    if (party >= Config::config->M)
    {
        MatrixXu temp;
        SocketUtil::receive(temp, 0);
        res = temp.cast<u128>();
    }
    return res;
}

Matrix128 FieldShare::reveal_for_nparty()
{
    int row = eta.rows();
    int col = eta.cols();
    vector<Matrix128> shares(Config::config->M);
    MatrixXu data_field = eta.cast<u64>();
    Matrix128 res = Matrix128::Zero(row, col);
    // res.setZero();
    for (int i = 0; i < Config::config->M; i++)
    {
        if (i == party)
        {
            shares[i] = eta;
        }
        else
        {
            SocketUtil::send(data_field, i);
        }
    }
    for (int i = 0; i < Config::config->M; i++)
    {
        if (i != party)
        {
            MatrixXu share_field;
            SocketUtil::receive(share_field, i);
            shares[i] = share_field.cast<u128>();
        }
    }
    for (int i = 0; i < Config::config->M; i++)
    {
        res += shares[i] * Mat::inv_for_mul[i];
    }
    res = res.unaryExpr([](const u128 x)
                        { return x % P; });
    res = P + gamma.array() - res.array();
    res = res.unaryExpr([](const u128 x)
                        { return x % P; });

    return res;
}

FieldShare FieldShare::Mul(const FieldShare &fieldshare)
{
    int row = eta.rows();
    int col = fieldshare.cols();
    Matrix128 eta_xy = Mul_offline::mul_triple(eta, fieldshare.eta);
    Matrix128 eta_z = Mul_offline::Random_matrix_for_trunc(row, col);
    Matrix128 z_ = Mul_offline::trunc_m_pr(eta_z, Config::config->K, Config::config->DECIMAL_LENGTH);

    Matrix128 gamma_x_y = gamma * fieldshare.eta;
    Matrix128 x_gamma_y = eta * fieldshare.gamma;
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    gamma_z_share += eta_xy + eta_z;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal(gamma_z_share);
    gamma_z += gamma * fieldshare.gamma;
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });

    Mat::truncateMatrix128(gamma_z);
    FieldShare res(z_, gamma_z);
    return res;
}

FieldShare FieldShare::Mul_tran(const FieldShare &fieldshare)
{
    // Matrix128 gamma_tran = gamma.transpose();
    int col = fieldshare.cols();
    int row = gamma.cols();

    Matrix128 eta_xy = Mul_offline::mul_triple(eta.transpose(), fieldshare.eta);
    Matrix128 eta_z = Mul_offline::Random_matrix_for_trunc(row, col);
    Matrix128 z_ = Mul_offline::trunc_m_pr(eta_z, Config::config->K, Config::config->DECIMAL_LENGTH);

    Matrix128 gamma_x_y = gamma.transpose() * fieldshare.eta;
    Matrix128 x_gamma_y = eta.transpose() * fieldshare.gamma;
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    // gamma_x_y = gamma_x_y.unaryExpr([](const u128 x) { return x % P; });
    // x_gamma_y = x_gamma_y.unaryExpr([](const u128 x) { return x % P; });
    gamma_z_share += eta_xy + eta_z;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal(gamma_z_share);
    gamma_z = gamma_z + gamma.transpose() * fieldshare.gamma;
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });

    Mat::truncateMatrix128(gamma_z);
    FieldShare res(z_, gamma_z);
    return res;
}

FieldShare FieldShare::Tran_mul(const FieldShare &fieldshare)
{
    // Matrix128 gamma_tran = gamma.transpose();
    int row = gamma.rows();
    int col = fieldshare.rows();

    Matrix128 eta_xy = Mul_offline::mul_triple(eta, fieldshare.eta.transpose());
    Matrix128 eta_z = Mul_offline::Random_matrix_for_trunc(row, col);
    Matrix128 z_ = Mul_offline::trunc_m_pr(eta_z, Config::config->K, Config::config->DECIMAL_LENGTH);

    Matrix128 gamma_x_y = gamma * fieldshare.eta.transpose();
    Matrix128 x_gamma_y = eta * fieldshare.gamma.transpose();
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    // gamma_x_y = gamma_x_y.unaryExpr([](const u128 x) { return x % P; });
    // x_gamma_y = x_gamma_y.unaryExpr([](const u128 x) { return x % P; });
    gamma_z_share += eta_xy + eta_z;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal(gamma_z_share);
    gamma_z = gamma_z + gamma * fieldshare.gamma.transpose();
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });

    Mat::truncateMatrix128(gamma_z);
    FieldShare res(z_, gamma_z);
    return res;
}

FieldShare FieldShare::operator+(const FieldShare &fieldshare) const
{
    Matrix128 temp0, temp1;
    temp0 = eta + fieldshare.eta;
    temp1 = gamma + fieldshare.gamma;
    FieldShare res(temp0, temp1);
    // res.residual();
    return res;
}

FieldShare FieldShare::operator-(const FieldShare &fieldshare) const
{
    Matrix128 temp0, temp1;
    temp0 = P + this->eta.array() - fieldshare.eta.array();
    temp1 = P + this->gamma.array() - fieldshare.gamma.array();
    FieldShare res(temp0, temp1);
    // res.residual();
    return res;
}

FieldShare FieldShare::operator*(const FieldShare &fieldshare) const
{
    int row = fieldshare.rows();
    int col = fieldshare.cols();

    Matrix128 eta_xy = Mul_offline::cwise_triple(eta, fieldshare.eta);
    Matrix128 eta_z = Mul_offline::Random_matrix(row, col);
    // Matrix128 eta_z = Matrix128::Zero(row, col);
    // Matrix128 eta_xy = Matrix128::Zero(row, col);

    Matrix128 gamma_x_y = gamma.cwiseProduct(fieldshare.eta);
    Matrix128 x_gamma_y = eta.cwiseProduct(fieldshare.gamma);
    Matrix128 gamma_z_share = (gamma_x_y + x_gamma_y).unaryExpr([](const u128 x)
                                                                { return P - (x % P); });
    gamma_z_share += eta_xy + eta_z;
    gamma_z_share = gamma_z_share.unaryExpr([](const u128 x)
                                            { return x % P; });
    Matrix128 gamma_z = FieldShare::Mul_reveal(gamma_z_share);
    gamma_z += gamma.cwiseProduct(fieldshare.gamma);
    gamma_z = gamma_z.unaryExpr([](const u128 x)
                                { return x % P; });
    FieldShare res(eta_z, gamma_z);
    return res;
}

FieldShare &FieldShare::operator=(const FieldShare *fieldshare)
{
    this->eta = fieldshare->eta;
    this->gamma = fieldshare->gamma;
    return *this;
}

FieldShare &FieldShare::operator+=(const FieldShare &fieldshare)
{
    this->eta += fieldshare.eta;
    this->gamma += fieldshare.gamma;
    return *this;
}

FieldShare FieldShare::operator+(const Matrix128 &c)
{
    Matrix128 temp = gamma + c;
    // temp = temp.unaryExpr([](const u128 x) { return x % P; });
    FieldShare res(eta, temp);
    return res;
}

FieldShare FieldShare::operator-(const Matrix128 &c)
{
    int row = this->eta.rows();
    int col = this->eta.cols();
    Matrix128 temp = P + gamma.array() - c.array();
    // temp = temp.unaryExpr([](const u128 x) { return x % P; });
    FieldShare res(this->eta, temp);
    return res;
}

FieldShare FieldShare::operator*(const Matrix128 &c)
{
    Matrix128 temp0, temp1;
    temp0 = eta.cwiseProduct(c);
    temp1 = gamma.cwiseProduct(c);
    FieldShare res(temp0, temp1);
    res.residual();
    return res;
}

FieldShare FieldShare::operator+(const u128 c) const
{
    Matrix128 temp = gamma.array() + c;
    // temp = temp.unaryExpr([](const u128 x) { return x % P; });
    FieldShare res(eta, temp);
    return res;
}

FieldShare FieldShare::operator-(const u128 c) const
{
    Matrix128 temp = P + gamma.array() - c;
    // temp = temp.unaryExpr([](const u128 x) { return x % P; });
    FieldShare res(eta, temp);
    return res;
}

FieldShare FieldShare::operator<(const u128 c) const
{
    FieldShare res = ltz(*this - c, Config::config->K);
    res.residual();
    return res;
}

FieldShare FieldShare::operator*(const u128 c) const
{
    Matrix128 temp0, temp1;
    temp0 = this->eta * c;
    temp1 = this->gamma * c;
    FieldShare res(temp0, temp1);
    // res.residual();
    return res;
}

FieldShare FieldShare::operator*(const double d) const
{
    int row = this->eta.rows();
    int col = this->eta.cols();

    u128 learning_rate = (u128)(d * Config::config->IE);

#ifdef TEST_FOR_TOTAL_TIME

    // Constant::Clock *clock_trun;
    // clock_trun = new Constant::Clock(2);
    Matrix128 d_eta = this->eta * learning_rate;
    d_eta = d_eta.unaryExpr([](const u128 x)
                            { return x % P; });
    Matrix128 d_eta_ie = Mul_offline::trunc_m_pr(d_eta, Config::config->K, Config::config->DECIMAL_LENGTH);
    // cout << "trun pair：" << clock_trun->get() << endl;

#else
    // offline 生成d*eta/IE 的share值，此处为了测试online时间，设为0
    Matrix128 d_eta_ie = Matrix128::Zero(row, col);
#endif
    Matrix128 temp0, temp1;

    temp1 = this->gamma * learning_rate;
    temp1 = temp1.unaryExpr([](const u128 x)
                            { return x % P; });
    // temp0 = this->eta * learning_rate;
    // temp0 = temp0.unaryExpr([](const u128 x)
    //                         { return x % P; });
    Mat::truncateMatrix128(temp1);
    temp0 = d_eta_ie;
    FieldShare res(temp0, temp1);
    return res;
}