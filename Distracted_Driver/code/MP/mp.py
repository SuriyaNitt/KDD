from multiprocessing import Process, Lock, Array
import numpy as np
import time

num_itr = 10000000

#global ping
#global pong
#ping = np.empty((num_itr))
#pong = np.empty((num_itr))

def produce_ping(itr, nm_itr, ping_local):
    for i in range(nm_itr):
        ping_local[i] = itr
        
def consume_ping(nm_itr, ping_local):
    sum = 0
    for i in range(nm_itr):
        for j in range(1):
            sum += ping_local[i]
    print sum
    
def produce_pong(itr, nm_itr, pong_local):
    for i in range(nm_itr):
        pong_local[i] = itr

def consume_pong(nm_itr, pong_local):
    sum = 0
    for i in range(nm_itr):
        for j in range(1):
            sum += pong_local[i]
    print sum

def f(l, i):
    l.acquire()
    print('hello world', i)
    l.release()

if __name__ == '__main__':
    #lock = Lock()

    #for num in range(10):
    #    Process(target=f, args=(lock, num)).start()
    
    ping = Array('i', range(num_itr))
    pong = Array('i', range(num_itr))
    
    a = time.time()
        
    #produce_ping(0)    
    for i in range(9):
        if i % 2 == 0:
            print('Producing ping')
            pa = Process(target=produce_pong(i+1, num_itr, pong))
            pb = Process(target=produce_ping(i+1, num_itr, ping))
            #pb = Process(target=consume_ping())
            pa.start()
            pb.start()
        else:
            print('Producing pong')
            pa = Process(target=produce_ping(i+1, num_itr, ping))
            pb = Process(target=produce_pong(i+1, num_itr, pong))
            #pb = Process(target=consume_pong())
            pa.start()
            pb.start()
            
    #consume_pong()
            
    b = time.time()
    
    print('Time Taken:{}'.format(b-a))
    
    
    
    
    
