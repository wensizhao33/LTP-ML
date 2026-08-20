cmake_minimum_required(VERSION 3.15)
project(pmpl_npc)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O3 -fopenmp")
include_directories(config)   #加入json库的路径

find_package(emp-tool REQUIRED)
include_directories(${EMP-TOOL_INCLUDE_DIRS})

find_package(emp-ot REQUIRED)
include_directories(${EMP-OT_INCLUDE_DIRS})

set(JSON_SOURCE_FILES config/json/reader.h config/json/json_reader.cpp config/json/json_value.cpp config/json/json_writer.cpp)
add_library(jsoncpp SHARED ${JSON_SOURCE_FILES})

# set(SOURCE_FILES_BASE config/Config.hpp config/Config.cpp Constant.h Constant.cpp util/Mat.h util/Mat.cpp util/IOManager.h util/IOManager.cpp util/SocketOnline.h util/SocketOnline.cpp util/SocketManager.h util/SocketManager.cpp util/Offline.h util/Offline.cpp core/Secret_Mul.h core/Secret_Mul.cpp core/Boolean_Share.h core/Boolean_Share.cpp core/Secret_Cmp.h core/Secret_Cmp.cpp)
# set(SOURCE_FILES_BASE config/Config.hpp config/Config.cpp Constant.h Constant.cpp util/Mat.h util/Mat.cpp util/IOManager.h util/IOManager.cpp util/SocketUtil.h util/SocketUtil.cpp  util/Offline.h util/Offline.cpp core/Secret_Mul.h core/Secret_Mul.cpp core/Boolean_Share.h core/Boolean_Share.cpp core/Secret_Cmp.h core/Secret_Cmp.cpp)
# set(Machine_Learning_FILES ml/linear.h ml/linear.cpp ml/logistic.h ml/logistic.cpp ml/nn.h ml/nn.cpp)

set(SOURCE_FILES_BASE config/Config.hpp config/Config.cpp Constant.h Constant.cpp util/Mat.h util/Mat.cpp util/IOManager.h util/IOManager.cpp util/SocketUtil.h util/SocketUtil.cpp  core/Offline.h core/Offline.cpp share/FieldShare.h share/FieldShare.cpp core/FieldCmp.h core/FieldCmp.cpp)
set(Machine_Learning_FILES ml/linear.h ml/linear.cpp ml/logistic.h ml/logistic.cpp ml/nn.h ml/nn.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -maes")

add_executable(pmpl_npc main.cpp ${SOURCE_FILES_BASE} ${Machine_Learning_FILES})
add_executable(test test.cpp ${SOURCE_FILES_BASE} ${Machine_Learning_FILES})
target_link_libraries(pmpl_npc jsoncpp)
target_link_libraries(pmpl_npc ${EMP-TOOL_LIBRARIES})
target_link_libraries(pmpl_npc ${EMP-OT_LIBRARIES})
target_link_libraries(test jsoncpp)
target_link_libraries(test ${EMP-TOOL_LIBRARIES})
target_link_libraries(test ${EMP-OT_LIBRARIES})