#include "FieldCmp.h"

FieldShare ltz(const FieldShare &a, int k)
{
    return P - div2m(a, k, k - 1);
}

// m = k-1
FieldShare div2m(const FieldShare &a, int k, int m)
{
    FieldShare temp = mod2m(a, m, k);
    FieldShare res = (a - temp) * pow_2_i_inverse[m];
    res.residual();
    return res;
}

// m = k-1, k_bit = k
FieldShare mod2m(const FieldShare &share, int m, int k_bit)
{
    int row = share.rows();
    int col = share.cols();
    FieldShare temp;

    FieldShare r2 = FieldShare(row, col);
    FieldShare r1 = FieldShare(row, col);
    FieldShare **bits;
    bits = new FieldShare *[m];
    for (int i = 0; i < m; i++)
        bits[i] = new FieldShare();

    Cmp_offline::p_rand_m(m, &r2, &r1, bits);

    temp = pow_2_i[k_bit - 1] + share + pow_2_i[m] * r2 + r1;

    temp.residual();
    Matrix128 b = temp.reveal();
    b = b.unaryExpr([&m](const u128 &x)
                    { return x % pow_2_i[m]; });
    FieldShare u, res;
    u = bitLT(b, bits, m);
    res = b - r1 + pow_2_i[m] * u;

    res.residual();

    return res;
}

//k_bit = m = k-1
FieldShare bitLT(const Matrix128 &a, FieldShare **bitShares, int k_bit)
{
    int row = a.rows();
    int col = a.cols();
    FieldShare bit_xor[k_bit];
    Matrix128 a_bit[k_bit];
    FieldShare **temp = new FieldShare *[k_bit];
    FieldShare **suf_mul = new FieldShare *[k_bit];
    for (int i = 0; i < k_bit; i++)
    {
        a_bit[i] = a.unaryExpr([&i](const u128 &x)
                               { return (x >> i) & 1; });
        bit_xor[i] = a_bit[i] + *bitShares[i] - 2 * a_bit[i] * *bitShares[i];
        bit_xor[i].residual();

        temp[i] = new FieldShare(bit_xor[i] + 1);
        suf_mul[i] = new FieldShare();
        // *temp[i] = bit_xor[i] + 1;
    }
    sufMul(k_bit, temp, suf_mul);

    FieldShare temp2(row, col);
    for (int i = 0; i < k_bit - 1; i++)
    {
        temp2 += (1 - a_bit[i].array()) * (*suf_mul[i] - *suf_mul[i + 1]);
    }
    FieldShare s = (1 - a_bit[k_bit - 1].array()) * bit_xor[k_bit - 1] + temp2;
    s.residual();
    for (int i = 0; i < k_bit; i++)
    {
        delete temp[i];
        delete suf_mul[i];
    }
    delete[] temp;
    delete[] suf_mul;

    FieldShare res = mod2(s, k_bit);
    return res;
}

// k = k-1
void sufMul(int k, FieldShare **shares, FieldShare **res)
{
    int row = shares[0]->rows();
    int col = shares[0]->cols();
    FieldShare **w = new FieldShare *[k];
    FieldShare **z = new FieldShare *[k];
    FieldShare w_combine(k * row, col);
    for (int i = 0; i < k; i++)
    {
        w[i] = new FieldShare();
        z[i] = new FieldShare();
    }
    Cmp_offline::presufMul(k, row, col, w, &w_combine, z);

    FieldShare **temp_shares = new FieldShare *[k];
    FieldShare **temp_res = new FieldShare *[k];
    for (int i = 0; i < k; i++)
    {
        temp_shares[i] = shares[k - i - 1];
        temp_res[i] = res[k - i - 1];
    }

    Matrix128 m[k];
    Matrix128 m_combine;

    //////////// defalt
    FieldShare share_combine(k * row, col);
    for (int i = 0; i < k; i++)
    {
        share_combine.block2(i * row, 0, row, col, *temp_shares[i]);
    }
    // FieldShare temp_combine;
    // temp_combine = w_combine * share_combine;
    m_combine = w_combine.Mulpub(share_combine);
    for (int i = 0; i < k; i++)
    {
        m[i] = m_combine.block(i * row, 0, row, col);
    }

    //////////// for LAN nn
    // for (int i = 0; i < k; i++)
    // {
    //     // Matrix128 temp =  w[i] * (*temp_shares[i]);
    //     m[i] = w[i]->Mulpub(*temp_shares[i]);
    // }

    ////////////
    for (int i = 1; i < k; i++)
    {
        m[i] = m[i].cwiseProduct(m[i - 1]);
        m[i] = m[i].unaryExpr([](const u128 x)
                              { return x % P; });
    }
    *temp_res[0] = *temp_shares[0];

    for (int i = 1; i < k; i++)
    {
        *temp_res[i] = *z[i] * m[i];
    }

    for (int i = 0; i < k; i++)
    {
        delete z[i];
        delete w[i];
    }
    delete[] z;
    delete[] w;

    delete[] temp_res;
    delete[] temp_shares;
}

//k_bit = k-1
FieldShare mod2(const FieldShare &a, int k_bit)
{
    int r = a.rows();
    int c = a.cols();
    FieldShare r2(r, c);
    FieldShare r1(r, c);
    FieldShare **bits = new FieldShare *[1];
    bits[0] = new FieldShare();

    Cmp_offline::p_rand_m(1, &r2, &r1, bits);
    FieldShare temp = pow_2_i[k_bit - 1] + a + 2 * r2 + *bits[0];
    temp.residual();
    Matrix128 temp_plaintext = temp.reveal();
    temp_plaintext = temp_plaintext.unaryExpr([](const u128 x)
                                              { return x & 1; });

    FieldShare res = temp_plaintext + *bits[0] - 2 * temp_plaintext * *bits[0];
    res.residual();

    for (int i = 0; i < 1; i++)
        delete bits[i];
    delete[] bits;

    return res;
}

FieldShare ReLU(FieldShare &share)
{
    // FieldShare temp = share < 0;

    FieldShare cmp = 1 - (share < 0);
    cout << cmp.reveal() << endl;

    FieldShare res = share * cmp;
    return res;
}

FieldShare Sigmoid(FieldShare &share)
{
    int row = share.rows();
    int col = share.cols();
    FieldShare cp1 = share + Config::config->IE / 2;
    cp1.residual();
    // FieldShare cmp1 = cp1 < 0;

    FieldShare cp2 = share + (P - Config::config->IE / 2);
    cp2.residual();
    // FieldShare cmp2 = cp2 < 0;
    FieldShare cp_combine(2 * row, col);
    cp_combine.block2(0, 0, row, col, cp1);
    cp_combine.block2(row, 0, row, col, cp2);
    FieldShare cmp_combine = cp_combine < 0;
    FieldShare cmp1 = cmp_combine.block1(0, 0, row, col);
    FieldShare cmp2 = cmp_combine.block1(row, 0, row, col);

    FieldShare temp1 = (1 - cmp1) * cmp2 * cp1;
    FieldShare res = temp1 + ((1 - cmp2) * (u128)Config::config->IE);
    res.residual();
    return res;
}