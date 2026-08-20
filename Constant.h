#ifndef CONSTANT_H
#define CONSTANT_H

#ifndef UNIX_PLATFORM
#define UNIX_PLATFORM
#endif

#include <iostream>
#include "config/Config.hpp"
#include <cstdio>
#include <vector>
#include <fstream>
#include <queue>
#include <Eigen/Dense>
#include <random>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <limits>
#ifdef UNIX_PLATFORM
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#else
#include <winsock2.h>
#endif

#define P 100000000000000003ll

#define DEBUG
#ifdef DEBUG
#define DBGprint(...) printf(__VA_ARGS__)
#else
#define DBGprint(...)
#endif
// #define TEST_FOR_TOTAL_TIME

using namespace std;
using namespace chrono;

typedef uint64_t u64;
typedef __uint128_t u128;
typedef uint8_t u8; // 8 bit
typedef long long ll;

extern int party;

class Constant
{
public:
    static string getDateTime()
    {
        time_t t = std::time(nullptr);
        struct tm *now = localtime(&t);
        char buf[80];
        strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", now);
        return string(buf);
    }
    class Clock
    {
        int id;
        high_resolution_clock::time_point start;
        static u64 global_clock[101];

    public:
        static double get_clock(int id);
        static void print_clock(int id);
        Clock(int id) : id(id)
        {
            start = high_resolution_clock::now();
        };
        ~Clock()
        {
            high_resolution_clock::time_point end = high_resolution_clock::now();
            decltype(duration_cast<microseconds>(end - start)) time_span = duration_cast<microseconds>(end - start);
            global_clock[id] += time_span.count();
        };
        double get()
        {
            high_resolution_clock::time_point end = high_resolution_clock::now();
            decltype(duration_cast<microseconds>(end - start)) time_span = duration_cast<microseconds>(end - start);
            return time_span.count() * 1.0 * microseconds::period::num / microseconds::period::den;
        }
        void print()
        {
            high_resolution_clock::time_point end = high_resolution_clock::now();
            decltype(duration_cast<microseconds>(end - start)) time_span = duration_cast<microseconds>(end - start);
            DBGprint("duration: %f\n", time_span.count() * 1.0 * microseconds::period::num / microseconds::period::den);
            // printf("duration: %f\n", time_span.count() * 1.0 * microseconds::period::num / microseconds::period::den);
        }
    };
    class Util
    {
    public:
        static void int_to_char(char *&p, int u);
        static void u64_to_char(char *&p, u64 u);
        static int char_to_int(char *&p);
        static u64 char_to_u64(char *&p);
        static void int_to_header(char *p, int u);
        static int header_to_int(char *p);
        static u64 double_to_u64(double x);
        static double u64_to_double(u64 u); // unsigned to double, long(signed)
        static u128 double_to_u128(double x);
        static double char_to_double(char *&p);
        static double field_to_double(u128 u);
        static void field_to_bool(bool *bool_, u128 u);

        static u64 getu64(char *&p);
        static u128 getu128(char *&p);
        static int getint(char *&p);

        static u64 random_u64();
        static u8 random_u8();
        static u128 random_field();

        static u128 power(u128 a, u128 b);
        static u128 inverse(u128 a);
        static void pow_2_init();
        static vector<u64> edabits();      // default with 64 bits
        static vector<u64> edabits(u64 r); // default with 64 bits
    };
};
std::ostream &
operator<<(std::ostream &dest, u128 value);

#endif // ACCESSCONTROL_CONSTANT_H