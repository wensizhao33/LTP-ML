
# RuYi
A Configurable and Efficient Secure Multi-Party Learning Framework with Privileged Parties

This is a code for experimentation.

## Configure

**constant.json**

Configure the number of parties, training parameters (e.g. batch size), and machine learning models. 

"M" refers to the number of total parties, "np" refers to the number of privileged parties, "na" refers to the number of assistant parties, and "nd" refers to the number of assistant parties allowed to drop out. 
Note that, $np + na = M$ and $nd < na$.


## Compile

Compile the executable file

```shell
cd Ruyi
cmake .
make -j
```

### Run

Download the MNIST dataset to the ./data folder and delete the first line of the dataset.

Start $M+nd$ processes and input the party index, respectively. We take $M=3, np =1, na = 2, nd = 1$ as an example:

```shell
./pmpl_npc 0
```

```shell
./pmpl_npc 1
```

```shell
./pmpl_npc 2
```

```shell
./pmpl_npc 3
```

Note that the first time to run the code, we should secret share the raw dataset among parties before training. 

(Ruyi/util/IOManager.cpp line 262)

```c++
ifstream infile("data/mnist/mnist_train.csv");
load_data(infile, train_data, train_label, Config::config->N);
if (party == 0)
    secret_share(train_data, train_label, "train");
infile.close();

// ifstream infile("data/mnist/mnist_train_" + to_string(party) + ".csv");
// ifstream infile_delta("data/mnist/mnist_train_delta.csv");
// load_ss(infile, train_data, train_label, infile_delta, train_data_delta, train_label_delta, Config::config->N);
// infile.close();
// infile_delta.close();

ifstream intest("data/mnist/mnist_test.csv");
load_data(intest, test_data, test_label, Config::config->testN);
if (party == 0)
    secret_share(test_data, test_label, "test");
intest.close();
```


After secret sharing the raw data, load the secret shared data and then perform training.

(Ruyi/util/IOManager.cpp line 262)

```c++
// ifstream infile("data/mnist/mnist_train.csv");
// load_data(infile, train_data, train_label, Config::config->N);
// if (party == 0)
//     secret_share(train_data, train_label, "train");
// infile.close();

ifstream infile("data/mnist/mnist_train_" + to_string(party) + ".csv");
ifstream infile_delta("data/mnist/mnist_train_delta.csv");
load_ss(infile, train_data, train_label, infile_delta, train_data_delta, train_label_delta, Config::config->N);
infile.close();
infile_delta.close();

ifstream intest("data/mnist/mnist_test.csv");
load_data(intest, test_data, test_label, Config::config->testN);
// if (party == 0)
//     secret_share(test_data, test_label, "test");
intest.close();
```
