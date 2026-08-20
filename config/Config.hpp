#include <cstdio>
#include <fstream>
#include "json/json.h"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

class Config
{
public:
    static Config *config;
    static Config *init(std::string file_name);
    Config(Json::Value root) : M(root["M"].asInt()),
                               np(root["np"].asInt()),
                               na(root["na"].asInt()),
                               nd(root["nd"].asInt()),
                               K(root["K"].asInt()),
                               KAPPA(root["KAPPA"].asInt()),
                               B(root["B"].asInt()),
                               D(root["D"].asInt()),
                               N(root["N"].asInt()),
                               testN(root["testN"].asInt()),
                               ML(root["ML"].asInt()),
                               Ep(root["Ep"].asInt()),
                               IE(root["IE"].asInt()),
                               R(root["R"].asDouble()),
                               nLayer(root["nLayer"].asInt()),
                               hiddenDim(root["hiddenDim"].asInt()),
                               numClass(root["numClass"].asInt()),
                               BIT_LENGTH(root["BIT_LENGTH"].asInt()),
                               DECIMAL_LENGTH(root["DECIMAL_LENGTH"].asInt()),
                               BUFFER_MAX(root["BUFFER_MAX"].asInt()),
                               HEADER_LEN(root["HEADER_LEN"].asInt()),
                               HEADER_LEN_OPT(root["HEADER_LEN_OPT"].asInt()){};
    const int M;              // Number of parties
    const int np;             // Number of privileged parties
    const int na;             // Number of assistant parties
    const int nd;             // Number of droped parties
    const int K;              // Bit length of the secret value
    const int KAPPA;          // Secret parameter
    const int B;              // Batch size
    const int D;              // Dimension
    const int N;              // the number of train data
    const int testN;          // the number of test data
    const int ML;             // 0 for linear regression; 1 for logistic; 2 for nn
    const int Ep;             // Epoch
    const double R;           // learning rate
    const int nLayer;         // the number of layers
    const int hiddenDim;      // the dimension of hidden layer
    const int numClass;       // the number of class
    const int IE;             // A factor to turn decimal into integer. With this factor, a fix-point rational number can be transformed into integer.
    const int BIT_LENGTH;     // the length of bit
    const int DECIMAL_LENGTH; // the length of decimal part
    const int BUFFER_MAX;     // MAX buffer size for socket
    const int HEADER_LEN;     // 
    const int HEADER_LEN_OPT; // 
};
