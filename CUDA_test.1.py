import pycuda.autoinit
import pycuda.driver as drv
import time
import numpy


from pycuda.compiler import SourceModule
gpu_test_source = SourceModule("""
#include <curand_kernel.h>

extern "C"{
    __global__ void test_kernel()
    {
        return;
    }
}
""", no_extern_c=True)

test_kernel= gpu_test_source.get_function("test_kernel")

a = numpy.array(range(0,10000))
result = numpy.array([0])

start = time.time()

test_kernel(block=(1024,1,1),grid=(1,1))
end = time.time()
print(result[0])
print(end-start)