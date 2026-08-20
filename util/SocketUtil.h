#ifndef SOCKETUTIL_H
#define SOCKETUTIL_H

#include "unistd.h"
#include "Mat.h"
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>

class SocketUtil
{
public:
    // SocketUtil ();
    // ~SocketUtil();
    static vector<string> ips;

    static NetIO *io;
    static emp::IKNP<emp::NetIO> *send_ot;
    static emp::IKNP<emp::NetIO> *recv_ot;

    static void socket_init();
    static void wait_for_connect(int port);
    static void connect_others(string ip, int port, int myport);
    static void greet();

    static void send_n(int fd, char *data, int len);
    static void recv_n(int sock, char *data, int len);
    static void send_n(FILE *file, char *data, int len);
    static void recv_n(FILE *file, char *data, int len);

    // void send_array(int pid, const Int_128_Array &mat);
    static void send(const MatrixXu &mat, int pid);
    static void send(const Matrix8 &mat, int pid);

    // void send_ring_array(int pid, const Ring_Array &mat);
    // void send_128_array(int pid, const Int_128_Array &mat);

    // void recv_array(int pid, Int_128_Array &mat);
    static void receive(MatrixXu &mat, int pid);
    static void receive(Matrix8 &mat, int pid);

    // static void print();

    // Int_128_Array recv_matrix(int pid);
};

class IO
{
public:
    int pid;
    IO(int pid)
    {
        this->pid = pid;
    }

    void flush()
    {
        return;
    }
    void send_data(const void *data, size_t nbyte);
    void recv_data(void *data, size_t nbyte);

    void send_block(const block *data, size_t nblock)
    {
        send_data(data, nblock * sizeof(block));
    }

    void recv_block(block *data, size_t nblock)
    {
        recv_data(data, nblock * sizeof(block));
    }

    void send_pt(Point *A, size_t num_pts = 1);

    void recv_pt(Group *g, Point *A, size_t num_pts = 1);
};

#endif // SOCKETUTIL_H
