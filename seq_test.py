import numpy
import time
b = numpy.array(range(0,10000000))
a = range(0,10000000)
result=0
start = time.time()
for i in a:
    result+=i
end = time.time()


print(result)
print(end-start)