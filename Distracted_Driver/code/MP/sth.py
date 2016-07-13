import numpy as np
import time 

num_itr = 10000000

global ping
global pong
ping = np.empty((num_itr))
pong = np.empty((num_itr))

def produce_ping(itr):
    global ping
    for i in range(num_itr):
        ping[i] = itr
        
def consume_ping():
    sum = 0
    for i in range(num_itr):
        for j in range(1):
            sum += ping[i]
    print sum
    
def produce_pong(itr):
    global pong
    for i in range(num_itr):
        pong[i] = itr

def consume_pong():
    sum = 0
    for i in range(num_itr):
        for j in range(1):
            sum += pong[i]
    print sum

a = time.time()

for i in range(10):
    if i%2 == 0:
        produce_pong(i)
        consume_pong()
    else:
        produce_ping(i)
        consume_ping()

b = time.time()

print('Time taken: {}'.format(b-a))
