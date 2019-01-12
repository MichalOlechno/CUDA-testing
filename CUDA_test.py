import pycuda.autoinit
import pycuda.driver as drv
import time
import numpy


from pycuda.compiler import SourceModule
mod = SourceModule("""

#include <curand_kernel.h>
    const int nstates = %(NGENERATORS)s;
    __device__ curandState_t* states[nstates];
extern "C"{
    __global__ void initkernel(int seed)
    {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates) {
            curandState_t* s = new curandState_t;
            if (s != 0) {
                curand_init(seed, tidx, 0, s);
            }
            states[tidx] = s;
            return;
        }
    }
}
""",nvcc="None",options="None",keep=False, no_extern_c=True)

#GPU_fill_rand= mod.get_function("GPU_fill_rand")
#generate_in_a_b= mod.get_function("generate_in_a_b")
init_func=mod.get_function("initkernel")
a = numpy.array(range(0,10000)).astype(numpy.float32)
#a_gpu = drv.mem_alloc(a.nbytes)
#drv.memcpy_htod(a_gpu, a)
seed = numpy.int32(123456789)
init_func(drv.In(seed), block=(1024,1,1), grid=(1,1,1))

result = numpy.array([0])

start = time.time()
#GPU_fill_rand(drv.Out(a),drv.In(numpy.int32(len(a))),block=(1024,1,1),grid=(1,1))
#generate_in_a_b(drv.Out(a),drv.In(numpy.int32(12)),drv.In(numpy.int32(64)),drv.In(numpy.int32(len(a))),block=(1024,1,1),grid=(1,1))
end = time.time()
print(result[0])
print(end-start)