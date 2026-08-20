#include "Constant.h"

int party;
u128 pow_2_i[64];
u128 pow_2_i_inverse[64];
u64 Constant::Clock::global_clock[101] = {0};

void Constant::Clock::print_clock(int id)
{
    DBGprint("clock: %d %f\n", id, global_clock[id] * 1.0 * microseconds::period::num / microseconds::period::den);
}

double Constant::Clock::get_clock(int id)
{
    return global_clock[id] * 1.0 * microseconds::period::num / microseconds::period::den;
}

void Constant::Util::int_to_char(char *&p, int u)
{
    *p++ = u & 0xff;
    *p++ = u >> 8 & 0xff;
    *p++ = u >> 16 & 0xff;
    *p++ = u >> 24 & 0xff;
}

void Constant::Util::u64_to_char(char *&p, u64 u)
{
    *p++ = u & 0xff;
    *p++ = u >> 8 & 0xff;
    *p++ = u >> 16 & 0xff;
    *p++ = u >> 24 & 0xff;
    *p++ = u >> 32 & 0xff;
    *p++ = u >> 40 & 0xff;
    *p++ = u >> 48 & 0xff;
    *p++ = u >> 56 & 0xff;
}

int Constant::Util::char_to_int(char *&p)
{
    int ret = 0;
    ret = *p++ & 0xff;
    ret |= (*p++ & 0xff) << 8;
    ret |= (*p++ & 0xff) << 16;
    ret |= (*p++ & 0xff) << 24;
    return ret;
}

u64 Constant::Util::char_to_u64(char *&p)
{
    u64 ret = 0;
    ret = (u64)(*p++ & 0xff);
    ret |= (u64)(*p++ & 0xff) << 8;
    ret |= (u64)(*p++ & 0xff) << 16;
    ret |= (u64)(*p++ & 0xff) << 24;
    ret |= (u64)(*p++ & 0xff) << 32;
    ret |= (u64)(*p++ & 0xff) << 40;
    ret |= (u64)(*p++ & 0xff) << 48;
    ret |= (u64)(*p++ & 0xff) << 56;
    return ret;
}

void Constant::Util::int_to_header(char *p, int u)
{
    *p++ = u & 0xff;
    *p++ = u >> 8 & 0xff;
    *p++ = u >> 16 & 0xff;
    *p++ = u >> 24 & 0xff;
}

int Constant::Util::header_to_int(char *p)
{
    int ret = 0;
    ret = *p++ & 0xff;
    ret |= (*p++ & 0xff) << 8;
    ret |= (*p++ & 0xff) << 16;
    ret |= (*p++ & 0xff) << 24;
    return ret;
}

u64 Constant::Util::double_to_u64(double x)
{
    return static_cast<u64>((long)(x * Config::config->IE));
}

double Constant::Util::u64_to_double(u64 u)
{
    return (long)u / (double)Config::config->IE;
}

u128 Constant::Util::double_to_u128(double x)
{
    return static_cast<u128>((ll)(x * Config::config->IE));
}

double Constant::Util::field_to_double(u128 u)
{
    ll temp;
    if (u > P / 2)
        temp = (ll)(u - P);
    else
        temp = (ll)u;
    double res;
    res = (double)(temp * 1.0) / Config::config->IE;
    return res;
}

double Constant::Util::char_to_double(char *&p)
{
    return strtod(p, NULL);
}

void Constant::Util::field_to_bool(bool *bool_, u128 u)
{
    for (int i = 0; i < 64; i++)
    {
        bool_[i] = (u & 1);
        u >>= 1;
    }
}

u64 Constant::Util::getu64(char *&p)
{
    while (!isdigit(*p))
        p++;
    u64 ret = 0;
    while (isdigit(*p))
    {
        ret = 10 * ret + *p - '0';
        p++;
    }
    return ret;
    //    bool negative = false;
    //     while (!isdigit(*p) && (*p != '-'))
    //         p++;
    //     u64 ret = 0;
    //     while (isdigit(*p) || (*p == '-'))
    //     {
    //         if (isdigit(*p))
    //         {
    //             ret = 10 * ret + *p - '0';
    //         }
    //         else
    //             negative = true;
    //         p++;
    //     }
    //     if (negative == true)
    //         return (-1) * ret;
    //     else
    //         return ret;
}

u128 Constant::Util::getu128(char *&p)
{
    while (*p && !isdigit(*p))
        p++;
    u128 ret = 0;
    while (*p && isdigit(*p))
    {
        ret = 10 * ret + *p - '0';
        p++;
    }
    return ret;
}

int Constant::Util::getint(char *&p)
{
    while (!isdigit(*p))
        p++;
    int ret = 0;
    while (isdigit(*p))
    {
        ret = 10 * ret + *p - '0';
        p++;
    }
    return ret;
}


u64 Constant::Util::random_u64()
{
    u64 ra = (u64)((rand()));
    u64 rb = (u64)((rand()));
    rb <<= 31;
    u64 rc = (u64)((rand())) % int(pow(2, 2));
    rc <<= 62;
    return (u64)((rc | rb | ra));
    // std::random_device rd;
    // std::mt19937_64 gen(rd());
    // std::uniform_int_distribution<u64> dis(std::numeric_limits<u64>::min(), std::numeric_limits<u64>::max());
    // return dis(gen);
}

u128 Constant::Util::random_field()
{
    u128 ra = (u128)((rand()));
    u128 rb = (u128)((rand()));
    rb <<= 31;
    u128 rc = (u128)((rand())) % int(pow(2, 2));
    rc <<= 62;
    u128 r_ring = (u128)(rc | rb | ra);
    return r_ring % P;
    // return 0;
}

u8 Constant::Util::random_u8()
{
    return rand() % 512;
}
// u64 secrandom_u64() {
//     std::random_device rd{};
//     std::mt19937 gen{rd()};
//     std::uniform_int_distribution<u64> dist;
//     return dist(gen);
// }

u128 Constant::Util::power(u128 a, u128 b)
{
    u128 res = 1;
    if (b == 0)
        return res;
    while (b > 0)
    {
        if (b & 1)
            res = res * a;
        res = res % P;
        a = a * a;
        a = a % P;
        b = b >> 1;
    }
    return res;
}

u128 Constant::Util::inverse(u128 a)
{
    return power(a, P - 2);
}

void Constant::Util::pow_2_init()
{
    for (int i = 0; i<64; i++)
    {
        pow_2_i[i] = (pow(2, i));
    }
    for (int i = 0; i < 64; i++){
        pow_2_i_inverse[i] = Constant::Util::inverse(pow_2_i[i]);
    }
}

std::ostream &
operator<<(std::ostream &dest, u128 value)
{
    std::ostream::sentry s(dest);
    if (s)
    {
        __uint128_t tmp = value < 0 ? -value : value;
        char buffer[128];
        char *d = std::end(buffer);
        do
        {
            --d; 
            *d = "0123456789"[tmp % 10];
            tmp /= 10;
        } while (tmp != 0);
        if (value < 0)
        {
            --d;
            *d = '-';
        }
        int len = std::end(buffer) - d;
        if (dest.rdbuf()->sputn(d, len) != len)
        {
            dest.setstate(std::ios_base::badbit);
        }
    }
    return dest;
}

