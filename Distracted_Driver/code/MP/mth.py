from threading import Thread, Lock
import numpy as np
import time
from multiprocessing import Process

num_itr = 12500000

global ping
global pong
ping = np.empty((num_itr))
pong = np.empty((num_itr))

ping_lock = Lock()
pong_lock = Lock()
master_lock = Lock()

global producing_ping
global producing_pong
global consuming_ping
global consuming_pong

producing_ping = 1
producing_pong = 0
consuming_ping = 0
consuming_pong = 1

def produce_ping(itr):
    global ping
    for i in range(num_itr):
        ping[i] = itr
        
def consume_ping():
    sum = 0
    for i in range(num_itr):
        sum += ping[i]
    print sum
    
def produce_pong(itr):
    global pong
    for i in range(num_itr):
        pong[i] = itr

def consume_pong():
    sum = 0
    for i in range(num_itr):
        sum += pong[i]
    print sum
        

def producer():
    global producing_ping
    global producing_pong
    global consuming_ping
    global consuming_pong
    itr = 0
    while itr < 19:
#        print ('Producer: {}'.format(producing_ping))
#        print ('Producer: {}'.format(producing_pong))
#        print ('Producer: {}'.format(consuming_ping))
#        print ('Producer: {}'.format(consuming_pong))
        if itr % 2 == 0:
            print('Producing Ping')
            while consuming_ping:
                pass
            ping_lock.acquire()
#            print('Inside producing ping')
            consuming_ping = 1
            produce_ping(itr)
            producing_ping = 0
            itr += 1
            ping_lock.release()
#            print('Produced Ping')
        else:
            print('Producing Pong')
            while consuming_pong:
                pass
            pong_lock.acquire()
#            print('Inside producing pong')
            consuming_pong = 1
            produce_pong(itr)
            producing_pong = 0
            itr += 1
            pong_lock.release()
#            print('Produced Pong')

def consumer():
    master_lock.acquire()
    global producing_ping
    global producing_pong
    global consuming_ping
    global consuming_pong
    itr = 0
    while itr < 20:
#        print ('Consumer: {}'.format(producing_ping))
#        print ('Consumer: {}'.format(producing_pong))
#        print ('Consumer: {}'.format(consuming_ping))
#        print ('Consumer: {}'.format(consuming_pong))
        if itr % 2 == 0:
            print('Consuming Pong')
            while producing_pong:
                pass
            pong_lock.acquire()
#            print('Inside consuming pong')
            producing_pong = 1
            consume_pong()
            consuming_pong = 0
            itr += 1
            pong_lock.release()
#            print('Consuming Pong')
        else:
            print('Consuming Ping')
            while producing_ping:
                pass
            ping_lock.acquire()
#            print('Inside consuming ping')
            producing_ping = 1
            consume_ping()
            consuming_ping = 0
            itr += 1
            ping_lock.release()
#            print('Consumed Ping')
    master_lock.release()

if __name__ == '__main__':
    a = time.time()
    
    for i in range(num_itr):
        pong[i] = -1    
    
    #producer_thread = Thread(target=producer)
    #consumer_thread = Thread(target=consumer)

    producer_thread = Process(target=producer)
    consumer_thread = Process(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    time.sleep(1)

    master_lock.acquire()
    b = time.time()
    print('Time Taken: {}'.format(b - a - 1))
    master_lock.release()

