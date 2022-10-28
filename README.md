# CUDA Weaver

A program that makes art with the power of Cuda.

![output](https://user-images.githubusercontent.com/21147581/193716579-c1791313-c733-484b-aa25-2aecae646678.png)

Usage:
```bash
./weaver [input.png] [ouput.png] -p [number of points] -r [resolution] -i [max iterations] -l [line thickness] -b [blur radius] -c [Colors like 'FF0000,FF6800,000055,000000,FFAA88']
```

## To compile

```
mkdir build
cd build
cmake ../source
make
```
