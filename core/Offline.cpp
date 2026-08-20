#include "Offline.h"

emp::PRG prg;

u64 extract_lo64(__m128i x)
{ // extract lower 64 bits of a block
    return (u64)_mm_cvtsi128_si64(x);
}

u64 extract_hi64(__m128i x)
{ // extract higher 64 bits of a block
    u64 *v64val = (u64 *)&x;
    return (u64)v64val[1];
}

void int_to_bool(bool *bool_, u64 int_)
{
    for (int i = 0; i < 58; i++)
    {
        bool_[i] = (int_ & 1);
        int_ >>= 1;
    }
}

Matrix128 Mul_offline::Reshare(Matrix128 x)
{
    int r = x.rows();
    int c = x.cols();
    Matrix128 res(r, c);
    res.setConstant(0);
    vector<Matrix128> recv(Config::config->M);
    vector<MatrixXu> x_share(Config::config->M + Config::config->nd);
    for (int i = 0; i < Config::config->M + Config::config->nd; i++)
    {
        x_share[i].resize(r, c);
    }
    if (party < Config::config->M)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                Matrix128 temp_vector(Config::config->M, 1);
                temp_vector(0, 0) = x(i, j);
                for (int k = 1; k < Config::config->M; k++)
                {
                    temp_vector(k, 0) = Constant::Util::random_field();
                }
                for (int k = 0; k < Config::config->M + Config::config->nd; k++)
                {
                    Matrix128 temp = Mat::PM_plus.row(k) * temp_vector;
                    temp = temp.unaryExpr([](const u128 x)
                                          { return x % P; });
                    x_share[k](i, j) = static_cast<u64>(temp(0, 0));
                }
            }
        }
    }

    if (party < Config::config->M)
    {
        for (int i = 0; i < Config::config->M + Config::config->nd; i++)
        {
            if (i == party)
            {
                recv[i] = x_share[i].cast<u128>();
            }
            else
            {
                SocketUtil::send(x_share[i], i);
            }
        }
        for (int i = 0; i < Config::config->M; i++)
        {
            if (i != party)
            {
                MatrixXu recv_field;
                SocketUtil::receive(recv_field, i);
                recv[i] = recv_field.cast<u128>();
            }
        }
    }
    else
    {
        for (int i = 0; i < Config::config->M; i++)
        {
            MatrixXu recv_field;
            SocketUtil::receive(recv_field, i);
            recv[i] = recv_field.cast<u128>();
        }
    }

    for (int i = 0; i < Config::config->M; i++)
    {
        res += recv[i];
    }
    res = res.unaryExpr([](const u128 x)
                        { return x % P; });
    return res;
}

Matrix128 Mul_offline::Random_matrix_for_trunc(int row, int col)
{
#ifdef TEST_FOR_TOTAL_TIME
    Matrix128 r = Mat::randomMatrixField(row, col);
    r = r.unaryExpr([](const u128 x)
                    { return (P + (x % Config::config->K) - Config::config->K / 2) % P; });
    return Mul_offline::Reshare(r);
#else
    Matrix128 r = Matrix128::Zero(row, col);
    return r;
#endif
}

Matrix128 Mul_offline::Random_matrix(int row, int col)
{
#ifdef TEST_FOR_TOTAL_TIME
    Matrix128 r = Mat::randomMatrixField(row, col);
    return Mul_offline::Reshare(r);
#else
    Matrix128 r = Matrix128::Zero(row, col);
    return r;
#endif
}

u128 Mul_offline::pre_mul_u128(u128 &a, u128 &b, int pid1, int pid2)
{

    int field_bit = 58;

    int num_ot = field_bit;

    // cout << num_ot << endl;
    // u128 ***r;

    u64 r_64[field_bit];
    // u64 sum_r;
    u128 sum = 0;

    for (int i = 0; i < field_bit; i++)
    {
        r_64[i] = Constant::Util::random_field();
        sum += (u128)r_64[i];
    }
    sum = sum % P;
    // sum_r = static_cast<u64>(sum);
    bool bits_B[field_bit];

    block *x0, *x1, *rec;

    x0 = new block[num_ot];
    x1 = new block[num_ot];
    rec = new block[num_ot];

    u128 randomA;
    u64 randomA_64;

    int_to_bool(bits_B, b);
    for (int i = 0; i < field_bit; i++)
    {
        randomA = ((a << i) + (u128)r_64[i]) % P;
        randomA_64 = static_cast<u64>(randomA);

        x0[i] = makeBlock(r_64[i], (u64)0);
        x1[i] = makeBlock(randomA_64, (u64)0);
    }

    if (party == pid1)
    {
        IO io(pid2);
        emp::IKNP<IO> ot1(&io), ot2(&io);
        ot1.send(x0, x1, num_ot);
        ot2.recv(rec, bits_B, num_ot);
    }
    else if (party == pid2)
    {
        IO io(pid1);
        emp::IKNP<IO> ot1(&io), ot2(&io);
        ot2.recv(rec, bits_B, num_ot);
        ot1.send(x0, x1, num_ot);
    }

    u128 temp = 0, c;
    for (int i = 0; i < field_bit; i++)
    {
        u64 temp_hi;
        temp_hi = extract_hi64(rec[i]);
        u128 temp_hi_128 = static_cast<u128>(temp_hi);
        temp += temp_hi_128;
    }
    c = Mat::coe(pid1, pid2) * (P - sum + temp) % P;
    return c;
}

Matrix128 Mul_offline::pre_mul_matrix(Matrix128 &a, Matrix128 &b, int pid1, int pid2)
{
    int row = a.rows();
    int col = a.cols();

    int field_bit = 58;

    int num_ot = (row + 1) / 2 * col * field_bit;
    int num_ot_per_bi = (row + 1) / 2;

    // cout << num_ot << endl;
    // u128 ***r;

    u64 ***r_64;
    Matrix128 sum_r(row, 1);
    r_64 = new u64 **[row];
    for (int i = 0; i < row; i++)
    {
        u128 sum = 0;
        r_64[i] = new u64 *[field_bit];
        for (int j = 0; j < field_bit; j++)
        {
            r_64[i][j] = new u64[col];
            prg.random_data(r_64[i][j], col * 8);
            for (int k = 0; k < col; k++)
            {
                sum += (u128)r_64[i][j][k];
            }
        }
        sum = sum % P;
        sum_r(i, 0) = sum;
    }

    bool bits_B[field_bit];
    bool select[num_ot];
    int index_select = 0;

    block *x0, *x1, *rec;
    int index = 0;

    x0 = new block[num_ot];
    x1 = new block[num_ot];
    rec = new block[num_ot];

    u64 temp_x_hi, temp_x_lo, temp_r_hi, temp_r_lo;

    u128 randomA[row];
    u64 randomA_64[row];

    for (int i = 0; i < col; i++)
    {
        int_to_bool(bits_B, b(i, 0));
        for (int j = 0; j < field_bit; j++)
        {
            for (int k = 0; k < (row + 1) / 2; k++)
                select[index_select++] = bits_B[j];

            for (int k = 0; k < row; k++)
            {
                randomA[k] = ((a(k, i) << j) + (u128)r_64[k][j][i]) % P;
                // randomA[k] = ((a(k, i) << j) + (u128)r_64[i][j][k]);
                randomA_64[k] = static_cast<u64>(randomA[k]);
            }
            for (int k = 0; k < row; k += 2)
            {
                if (row - k == 1)
                {
                    temp_r_hi = r_64[k][j][i];
                    temp_r_lo = (u64)0;
                    temp_x_hi = randomA_64[k];
                    temp_x_lo = (u64)0;
                }
                else
                {
                    temp_r_hi = r_64[k][j][i];
                    temp_r_lo = r_64[k + 1][j][i];
                    temp_x_hi = randomA_64[k];
                    temp_x_lo = randomA_64[k + 1];
                }
                x0[index] = makeBlock(temp_r_hi, temp_r_lo);
                x1[index] = makeBlock(temp_x_hi, temp_x_lo);
                index++;
            }
        }
    }

    if (party == pid1)
    {
        IO io(pid2);
        emp::IKNP<IO> ot1(&io), ot2(&io);
        ot1.send(x0, x1, num_ot);
        ot2.recv(rec, select, num_ot);
    }
    else if (party == pid2)
    {
        IO io(pid1);
        emp::IKNP<IO> ot1(&io), ot2(&io);
        ot2.recv(rec, select, num_ot);
        ot1.send(x0, x1, num_ot);
    }

    Matrix128 temp(row, 1);
    temp.setZero();
    int index_rec = 0;
    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < field_bit; j++)
        {
            int index_c = 0;
            for (int k = 0; k < num_ot_per_bi; k++)
            {
                u128 temp_hi_128, temp_lo_128;
                temp_hi_128 = static_cast<u128>(extract_hi64(rec[index_rec]));
                temp(index_c, 0) = (temp(index_c, 0) + temp_hi_128);

                index_c++;
                if (index_c < row)
                {
                    temp_lo_128 = static_cast<u128>(extract_lo64(rec[index_rec]));
                    temp(index_c, 0) = (temp(index_c, 0) + temp_lo_128);
                    index_c++;
                }
                index_rec++;
            }
        }
    }

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < field_bit; j++)
        {
            delete r_64[i][j];
        }
        delete r_64[i];
    }
    delete r_64;

    Matrix128 c;
    Matrix128 c_field(row, 1);
    c_field.setConstant(P);
    c = (c_field - sum_r + temp).unaryExpr([](const u128 x)
                                           { return x % P; });
    c = (Mat::coe(pid1, pid2) * c).unaryExpr([](const u128 x)
                                             { return x % P; });
    return c;
}

Matrix128 Mul_offline::mul_triple(Matrix128 a, Matrix128 b)
{
    Matrix128 res;
#ifdef TEST_FOR_TOTAL_TIME
    // Constant::Clock *clock_test;
    // clock_test = new Constant::Clock(1);
    Matrix128 ab(a.rows(), b.cols());

    if (party < Config::config->M)
    {
        ab = (Mat::coe(party, party) * a * b).unaryExpr([](const u128 x)
                                                        { return x % P; });
        for (int d = 0; d < b.cols(); d++)
        {
            Matrix128 b_col = b.col(d);

            for (int i = 0; i < Config::config->M; i++)
            {
                for (int j = i + 1; j < Config::config->M; j++)
                {
                    if ((party == i || party == j))
                    {
                        ab.col(d) += pre_mul_matrix(a, b_col, i, j);
                    }
                }
            }
        }
    }

    ab = ab.unaryExpr([](const u128 x)
                      { return x % P; });

    // cout << "online time:" << clock_test->get() << endl;

    // Constant::Clock *clock_test2;
    // clock_test2 = new Constant::Clock(1);

    res = Mul_offline::Reshare(ab);

    // cout << "online time:" << clock_test2->get() << endl;
#else
    res = Matrix128::Zero(a.rows(), b.cols());
#endif
    return res;
}

Matrix128 Mul_offline::cwise_triple(Matrix128 a, Matrix128 b)
{
    int row = a.rows(), col = a.cols();
    Matrix128 res(row, col);
    Matrix128 ab(row, col);
#ifdef TEST_FOR_TOTAL_TIME
    if (party < Config::config->M)
    {
        Matrix128 ab1 = Mat::coe(party, party) * a.cwiseProduct(b);
        ab = ab1.unaryExpr([](const u128 x)
                           { return x % P; });
        for (int r = 0; r < row; r++)
        {
            for (int c = 0; c < col; c++)
            {
                for (int i = 0; i < Config::config->M; i++)
                {
                    for (int j = i + 1; j < Config::config->M; j++)
                    {
                        if ((party == i || party == j))
                        {
                            ab(r, c) += pre_mul_u128(a(r, c), b(r, c), i, j);
                        }
                    }
                }
            }
        }
    }

    ab = ab.unaryExpr([](const u128 x)
                      { return x % P; });

    res = Mul_offline::Reshare(ab);
#else
    res.setZero();
#endif
    return res;
}

Matrix128 Mul_offline::cwise_triple_nparty(Matrix128 a, Matrix128 b)
{
    int row = a.rows(), col = a.cols();
    Matrix128 ab(row, col);
#ifdef TEST_FOR_TOTAL_TIME
    Matrix128 ab1 = Mat::coe(party, party) * a.cwiseProduct(b);
    ab = ab1.unaryExpr([](const u128 x)
                       { return x % P; });
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < col; c++)
        {
            for (int i = 0; i < Config::config->M; i++)
            {
                for (int j = i + 1; j < Config::config->M; j++)
                {
                    if ((party == i || party == j))
                    {
                        ab(r, c) += pre_mul_u128(a(r, c), b(r, c), i, j);
                    }
                }
            }
        }
    }
    ab = ab.unaryExpr([](const u128 x)
                      { return x % P; });
    ab = (ab * Mat::inv_for_div[party]).unaryExpr([](const u128 x)
                                                  { return x % P; });
#else
    ab.setZero();
#endif
    return ab;
}

Matrix128 Mul_offline::p_rand_int(int r, int c, int bit)
{
    Matrix128 res(r, c);
    // res.setZero();
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            res(i, j) = random() % pow_2_i[bit];
        }
    }
    return res;
}

Matrix128 Mul_offline::p_rand_bit(int r, int c)
{
    Matrix128 R(r, c);
    R = Mat::randomMatrixField(r, c);
    Matrix128 u(r, c);
    u = FieldShare::Mulpub(R);
    u = FieldShare::Mul_reveal_for_nparty(u);
    Matrix128 v(r, c);
    v = Mat::power_matrix(u, (P + 1) / 4);
    v = Mat::inverse_matrix(v);

    Matrix128 b(r, c);
    Matrix128 one(r, c);
    one.setConstant(1);

    b = v.cwiseProduct(R) + one;
    b = b.unaryExpr([](const u128 x)
                    { return x % P; });
    b = b * pow_2_i_inverse[1];

    b = b.unaryExpr([](const u128 x)
                    { return x % P; });
    return b;
}

// r1 = r'  r2 = r''
void Mul_offline::p_rand_m(int k, int m, Matrix128 &r1, Matrix128 &r2, Matrix128 **bits)
{
    int r = r1.rows();
    int c = r1.cols();
    r2 = p_rand_int(r, c, k + Config::config->KAPPA - m);
    for (int i = 0; i < m; i++)
    {
        *bits[i] = p_rand_bit(r, c);
        r1 += pow_2_i[i] * (*bits[i]);
    }
    r1 = r1.unaryExpr([](const u128 x)
                      { return x % P; });
}

// m 即小数位数，k 即整数位数
Matrix128 Mul_offline::trunc_m_pr(Matrix128 eta, int k, int m)
{
    int r = eta.rows();
    int c = eta.cols();
    Matrix128 d(r, c);
#ifdef TEST_FOR_TOTAL_TIME
    if (party < Config::config->M)
    {

        Matrix128 r1(r, c), r2(r, c);
        // r1 = r'  r2 = r''
        r1.setZero();
        r2.setZero();
        Matrix128 **bits = new Matrix128 *[m];
        for (int i = 0; i < m; i++)
        {
            bits[i] = new Matrix128();
        }
        p_rand_m(k, m, r1, r2, bits);

        Matrix128 temp = pow_2_i[k - 1] + eta.array() + (pow_2_i[m] * r2).array() + r1.array();
        temp = temp.unaryExpr([](const u128 x)
                              { return x % P; });
        Matrix128 temp_plaintext = FieldShare::Mul_reveal_for_nparty(temp);

        temp_plaintext = temp_plaintext.unaryExpr([&m](const u128 x)
                                                  { return x % pow_2_i[m]; });
        d = pow_2_i_inverse[m] * (P + eta.array() - temp_plaintext.array() + r1.array());
        d = d.unaryExpr([](const u128 x)
                        { return x % P; });
        d = (d * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                    { return x % P; });
    }
    return Mul_offline::Reshare(d);
#else
    d.setZero();
    return d;
#endif
}

Cmp_offline::Cmp_offline(int row, int col, int k, int m)
{
    r2 = FieldShare(row, col);
    r1 = FieldShare(row, col);
    bits = new FieldShare *[m];
    for (int i = 0; i < m; i++)
        bits[i] = new FieldShare();
    p_rand_m(m, &r2, &r1, bits);

    r = new FieldShare *[k];
    w = new FieldShare *[k];
    r_inverse = new FieldShare *[k];
    w_combine = FieldShare(m * row, col);
    for (int i = 0; i < m; i++)
    {
        r[i] = new FieldShare(row, col);
        w[i] = new FieldShare(row, col);
        r_inverse[i] = new FieldShare(row, col);

        r[i]->setConstant(1);
        w[i]->setConstant(1);
        r_inverse[i]->setConstant(1);

        w_combine.block2(i * row, 0, row, col, *w[i]);
    }
}

void Cmp_offline::p_rand_m(int m, FieldShare *r2, FieldShare *r1, FieldShare **bits, int k_bit)
{
    int r = r2->rows();
    int c = r2->cols();
    *r2 = p_rand_int(r, c, k_bit + Config::config->KAPPA - m);
    for (int i = 0; i < m; i++)
    {
        *bits[i] = p_rand_bit(r, c);
        *r1 = pow_2_i[i] * *bits[i] + *r1;
        r1->residual();
    }
}

FieldShare Cmp_offline::p_rand_int(int r, int c, int k)
{
#ifdef TEST_FOR_TOTAL_TIME
    FieldShare res(r, c);
    res.gamma.setConstant(0);
    if (party < Config::config->M)
    {
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                // res.eta(i, j) = P - (random() % pow_2_i[k]);
                res.eta(i, j) = (P - (random() % pow_2_i[k])) * Mat::inv_for_mul[party];
                res.eta(i, j) = res.eta(i, j) % P;
            }
        }
    }
    res.eta = Mul_offline::Reshare(res.eta);
    return res;
#else
    FieldShare res(r, c);
    res.setConstant(1);
    return res;
#endif
}

FieldShare Cmp_offline::p_rand_bit(int r, int c)
{
#ifdef TEST_FOR_TOTAL_TIME
    FieldShare b(r, c);
    if (party < Config::config->M)
    {
        FieldShare R(r, c);
        R = FieldShare::Random(r, c);
        // FieldShare u(r, c);
        // u = R.Mulpub_for_nparty(R);
        Matrix128 raw_u = R.Mulpub_for_nparty(R);
        Matrix128 v(r, c);
        v = Mat::power_matrix(raw_u, (P + 1) / 4);
        v = Mat::inverse_matrix(v);

        Matrix128 one(r, c);
        one.setConstant(1);
        b = v * R + one;
        b.residual();
        b = b * pow_2_i_inverse[1];
        b.residual();
        b.eta = (b.eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                            { return x % P; });
    }
    if (party == 0)
    {
        MatrixXu temp = b.gamma.cast<u64>();
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
        b.gamma = temp.cast<u128>();
    }
    b.eta = Mul_offline::Reshare(b.eta);
    return b;
#else
    FieldShare ran(r, c);
    ran.setConstant(0);
    return ran;
#endif
}

void Cmp_offline::presufMul(int k, int row, int col, FieldShare **w, FieldShare *w_combine, FieldShare **z)
{
#ifdef TEST_FOR_TOTAL_TIME
    FieldShare z_combine(k * row, col);
    if (party < Config::config->M)
    {
        FieldShare **r = new FieldShare *[k];
        for (int i = 0; i < k; i++)
        {
            r[i] = new FieldShare();
        }
        FieldShare s[k];
        FieldShare v[k];
        Matrix128 u[k];
        Matrix128 u_inverse[k];
        for (int i = 0; i < k; i++)
        {
            *r[i] = FieldShare::Random(row, col);
            s[i] = FieldShare::Random(row, col);
            // FieldShare temp = r[i]->Mulpub_for_nparty(s[i]);
            // u[i] = temp.reveal_for_nparty();
            u[i] = r[i]->Mulpub_for_nparty(s[i]);
            for (int j = 0; j < row * col; j++)
            {
                while (u[i](j) == 0)
                {
                    FieldShare x = FieldShare::Random(1, 1);
                    FieldShare y = FieldShare::Random(1, 1);
                    // temp = x.Mulpub_for_nparty(y);
                    // u[i](j) = temp.reveal_for_nparty()(0);
                    u[i](j) = x.Mulpub_for_nparty(y)(0);
                    r[i]->eta(j) = x.eta(0);
                    r[i]->gamma(j) = x.gamma(0);
                    s[i].eta(j) = y.eta(0);
                    s[i].gamma(j) = y.gamma(0);
                }
            }
            u_inverse[i] = Mat::inverse_matrix(u[i]);
        }
        for (int i = 0; i < k - 1; i++)
        {
            v[i] = r[i + 1]->Cwise_for_nparty(s[i]);
        }

        *w[0] = *r[0];
        w_combine->block2(0, 0, row, col, *w[0]);

        for (int i = 1; i < k; i++)
        {
            *w[i] = v[i - 1] * u_inverse[i - 1];
            w_combine->block2(i * row, 0, row, col, *w[i]);
        }
        for (int i = 0; i < k; i++)
        {
            *z[i] = s[i] * u_inverse[i];
            z_combine.block2(i * row, 0, row, col, *z[i]);
        }

        w_combine->eta = (w_combine->eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                                              { return x % P; });
        z_combine.eta = (z_combine.eta * Mat::inv_for_mul[party]).unaryExpr([](const u128 x)
                                                                            { return x % P; });

        for (int i = 0; i < k; i++)
        {
            delete r[i];
        }
        delete[] r;
    }

    if (party == 0)
    {
        MatrixXu temp_w = w_combine->gamma.cast<u64>();
        MatrixXu temp_z = z_combine.gamma.cast<u64>();
        for (int i = Config::config->M; i < Config::config->M + Config::config->nd; i++)
        {
            SocketUtil::send(temp_w, i);
            SocketUtil::send(temp_z, i);
        }
    }
    if (party >= Config::config->M)
    {
        MatrixXu temp_w;
        MatrixXu temp_z;
        temp_w.setZero();
        temp_z.setZero();
        SocketUtil::receive(temp_w, 0);
        SocketUtil::receive(temp_z, 0);
        w_combine->gamma = temp_w.cast<u128>();
        z_combine.gamma = temp_z.cast<u128>();
    }

    w_combine->eta = Mul_offline::Reshare(w_combine->eta);
    z_combine.eta = Mul_offline::Reshare(z_combine.eta);

    for (int i = 0; i < k; i++)
    {
        w[i]->gamma = w_combine->gamma.block(i * row, 0, row, col);
        w[i]->eta = w_combine->eta.block(i * row, 0, row, col);
        z[i]->gamma = z_combine.gamma.block(i * row, 0, row, col);
        z[i]->eta = z_combine.eta.block(i * row, 0, row, col);
    }

#else
    for (int i = 0; i < k; i++)
    {
        z[i]->resize(row, col);
        w[i]->resize(row, col);
        z[i]->setConstant(1);
        w[i]->setConstant(1);
    }
    w_combine->resize(k * row, col);
    w_combine->setConstant(1);
#endif
}
