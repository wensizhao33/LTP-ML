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
Cmp_offline *lor_cmp_off1;
Cmp_offline *lor_cmp_off2;
Cmp_offline *nn_cmp_off1;
Cmp_offline *nn_cmp_off2;
Cmp_offline *nn_cmp_off3;
Cmp_offline *nn_cmp_off4;

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

    // nn_cmp_off1 = new Cmp_offline(2, 2, Config::config->K, Config::config->K - 1);
    // nn_cmp_off3 = new Cmp_offline(2, 2, Config::config->K, 1);

    // FieldShare res = ReLU(share);
    // cout << res.reveal() << endl;
    // cout << "online time:" << clock_relu->get() << endl;
    // test_reveal();
    omp_set_num_threads(8);
    if (Config::config->ML == 0)
        Linear::train_model();
    else if (Config::config->ML == 1)
        Logistic::train_model();
    else if (Config::config->ML == 2)
        NN::train_model();
    // cout << "send size:" << send_size << endl
    //      << "receive size: " << receive_size << endl;
    // cout << "rounds: " << rounds << endl;
}