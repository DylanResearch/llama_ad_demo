cmake_minimum_required(VERSION 3.28)
project (llama_ad_demo)

set(CMAKE_CXX_STANDARD 17) #C++26
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # .exe, .dll
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)   # shared libs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)   # static libs / .lib import files

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(llama_ad_demo "${CMAKE_SOURCE_DIR}/main.cpp" ${PROJECT_SOURCES}) #create an executable from main.cpp

target_compile_definitions(llama_ad_demo PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}") #adding filepath for easier folder management
target_include_directories(llama_ad_demo PRIVATE src)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp llama)
add_subdirectory(llama.cpp/tools/mtmd mtmd)
add_subdirectory(llama.cpp/tools/tts tts)

target_link_libraries(llama_ad_demo PRIVATE llama) #linking so we can include llama.cpp headers in our code
target_link_libraries(llama_ad_demo PRIVATE mtmd)
target_link_libraries(llama_ad_demo PRIVATE gdiplus) #image preview in UI
target_link_libraries(llama_ad_demo PRIVATE winmm) #PlaySound for generated wav playback