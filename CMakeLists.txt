cmake_minimum_required(VERSION 3.28)
project (local_models_cpp)

set(CMAKE_CXX_STANDARD 17) #C++26
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # .exe, .dll
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)   # shared libs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)   # static libs / .lib import files

add_executable(local_models_cpp 
                main.cpp
                src/LlamaInterface.cpp) #create an executable from main.cpp

target_include_directories(local_models_cpp PRIVATE src) 
#target_include_directories(local_models_cpp PRIVATE llama.cpp/tools/mdmt)

add_subdirectory(llama.cpp llama)
add_subdirectory(llama.cpp/tools/mtmd mtmd)
target_link_libraries(local_models_cpp PRIVATE llama) #linking so we can include llama.cpp headers in our code
target_link_libraries(local_models_cpp PRIVATE mtmd) 