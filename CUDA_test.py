import pycuda.autoinit
import pycuda.driver as drv
import time
import numpy


from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void add_all(float *result, float *a,int params)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
   //printf("%d ",i);
   result[0] +=a[i];
}
""")

add_all= mod.get_function("add_all")

a = numpy.array(range(0,10000000))
result = numpy.array([0])
params= numpy.array([0])
params[0]=len(a)
print(params[0])
start = time.time()
add_all(drv.Out(result),drv.In(a),drv.In(numpy.int32(params[0])),block=(1024,1,1),grid=(20,20))
end = time.time()
print(result[0])
print(end-start)