import numpy
import time
import random as rn
#b = numpy.array(range(0,10000000))
#a = range(0,10000000)
#result=0
a = numpy.array(range(0,10240))
start = time.time()
#for i in a:
#    result+=i
R_chosen = rn.sample(range(0,1000000), 10240*5)
end = time.time()


#print(R_chosen)
print(end-start)