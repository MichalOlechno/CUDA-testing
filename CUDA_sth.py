import numpy as np
import time
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

code = """
    #include <curand_kernel.h>
    #include <math.h>  

    const int nstates = %(NGENERATORS)s;
    const int number = %(NUMBERS)s;
    const int maxNumber = %(MAXNUMBER)s;
    __device__ curandState_t* states[nstates*number];

    __global__ void initkernel(int seed)
    {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates*number) {
            curandState_t* s = new curandState_t;
            if (s != 0) {
                curand_init(seed, tidx, 0, s);
            }

            states[tidx] = s;
        }
    }

    __global__ void randfillkernel(float *values,float *inptdata, int N)
    {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates*number) {
            curandState_t s = *states[(tidx)];
            for(int i=tidx; i < N; i += blockDim.x * gridDim.x) {
                values[i] = curand_uniform(&s);
                values[i]= trunc(maxNumber*values[i]);
                
            }
            *states[tidx] = s;
        }
    }
"""

N = 1024
numbers=50
nvalues = 10240*5
mod = SourceModule(code % { "NGENERATORS" : N,"NUMBERS" : numbers,"MAXNUMBER" :nvalues}, no_extern_c=True)
init_func = mod.get_function("_Z10initkerneli")
fill_func = mod.get_function("_Z14randfillkernelPfi")

seed = np.int32(123456789)

init_func(drv.In(seed), block=(1024,1,1), grid=(50,1,1))
gdata = gpuarray.zeros(10240*5, dtype=np.float32)
#data = np.zeros(10240, dtype=np.float32)
start = time.time()
fill_func(gdata,gdata,drv.In(np.int32(nvalues)),drv.In(np.int32(nvalues)), block=(1024,1,1), grid=(50,1,1))
end = time.time()

#arr=list()
print(end-start)
#for i in gdata:
    #if i>0.0:
        #arr.append()
#print(len(arr))
print(gdata[1024*5])
print(gdata[10230*5])

