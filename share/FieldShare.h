#ifndef FIELDSHARE_H
#define FIELDSHARE_H

#include "../util/Mat.h"

extern int party;

class FieldShare
{
public:
    Matrix128 eta;
    Matrix128 gamma;
    FieldShare();
    FieldShare(int r, int c);
    FieldShare(const Matrix128 &eta, const Matrix128 &gamma);
    int size() const;
    int rows() const;
    int cols() const;
    void resize(int r, int c);
    void setZero();
    void setConstant(int c);
    FieldShare transpose();
    FieldShare block1(int startRow, int startCol, int blockRows, int blockCols);
    void block2(int startRow, int startCol, int blockRows, int blockCols, FieldShare &share);
    void residual();
    static FieldShare Random(int row, int col);
    Matrix128 Mulpub_for_nparty(const FieldShare &fieldshare);
    Matrix128 Mulpub(const FieldShare &x);
    FieldShare Cwise_for_nparty(const FieldShare &fieldshare);
    // 两个相同的数相乘
    static Matrix128 Mulpub(const Matrix128 &x);
    static MatrixXd decode(FieldShare &share);
    static Matrix128 Mul_reveal(const Matrix128 &data);
    static Matrix128 Mul_reveal_for_nparty(const Matrix128 &data);

    FieldShare truncation();
    Matrix128 reveal();
    Matrix128 reveal_for_nparty();

    FieldShare Mul(const FieldShare &fieldshare);
    FieldShare Mul_tran(const FieldShare &fieldshare);
    FieldShare Tran_mul(const FieldShare &fieldshare);
    FieldShare operator+(const FieldShare &share) const;
    FieldShare operator-(const FieldShare &share) const;
    FieldShare operator*(const FieldShare &share) const;
    FieldShare &operator=(const FieldShare *share);
    FieldShare &operator+=(const FieldShare &share);

    FieldShare operator+(const Matrix128 &c);
    FieldShare operator-(const Matrix128 &c);
    FieldShare operator*(const Matrix128 &c);

    FieldShare operator+(const u128 c) const;
    FieldShare operator-(const u128 c) const;
    FieldShare operator<(const u128 c) const;
    FieldShare operator*(const u128 c) const;
    FieldShare operator*(const double d) const;

    friend FieldShare operator+(const Matrix128 &c, const FieldShare &share)
    {
        Matrix128 temp = share.gamma + c;
        // temp = temp.unaryExpr([](const u128 x) { return x % P; });
        FieldShare res(share.eta, temp);
        return res;
    }

    friend FieldShare operator-(const Matrix128 &c, const FieldShare &share)
    {
        Matrix128 temp = P + c.array() - share.gamma.array();
        temp = temp.unaryExpr([](const u128 x)
                              { return x % P; });
        FieldShare res(P - share.eta.array(), temp);
        return res;
    }

    friend FieldShare operator*(const Matrix128 &c, const FieldShare &share)
    {
        Matrix128 temp0, temp1;
        temp0 = share.eta.cwiseProduct(c);
        temp1 = share.gamma.cwiseProduct(c);
        FieldShare res(temp0, temp1);
        res.residual();
        return res;
    }

    friend FieldShare operator+(const u128 c, const FieldShare &share)
    {
        Matrix128 temp = share.gamma.array() + c;
        // temp = temp.unaryExpr([](const u128 x) { return x % P; });
        FieldShare res(share.eta, temp);
        return res;
    }

    friend FieldShare operator-(const u128 c, const FieldShare &share)
    {
        Matrix128 temp = P + c - share.gamma.array();
        // temp = temp.unaryExpr([](const u128 x) { return x % P; });
        FieldShare res(P - share.eta.array(), temp);
        return res;
    }

    friend FieldShare operator*(const u128 c, const FieldShare &share)
    {
        Matrix128 temp0, temp1;
        temp0 = share.eta * c;
        temp1 = share.gamma * c;
        FieldShare res(temp0, temp1);
        // res.residual();
        return res;
    }
};

#endif // FIELDSHARE_H