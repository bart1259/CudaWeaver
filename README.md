# CUDA Weaver

A program that makes single thread weave art with the power of Cuda.

## Black and White
![output](https://user-images.githubusercontent.com/21147581/193716579-c1791313-c733-484b-aa25-2aecae646678.png)
## Color
![output](https://user-images.githubusercontent.com/21147581/202290839-16455be3-414e-419a-b154-a9d6e8b5f6f6.png)

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
