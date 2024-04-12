# How to build

```shell
cmake -S . -B ../build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
./build/bin/ltct
```