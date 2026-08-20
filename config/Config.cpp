#include "Config.hpp"

Config *Config::config = nullptr;
// jsonfile path
Config *Config::init(std::string file_name)
{
    if (Config::config == nullptr)
    {
        Json::Value root;
        Json::Reader reader;
        // open json file
        std::ifstream ifs(file_name);
        if (!reader.parse(ifs, root))
        {
            std::cout << "fail to parse" << std::endl;
        }
        Config::config = new Config(root);
    }
    return Config::config;
}