import simpy 
import numpy as np 

def customer(env, name, counter, mean_service_time):
    ## counter: 사용하는 리소스 
    ## mean_service_time: 서비스 시간 평균 
    arrive_time = env.now
    print('%7.4f %s: Here I am' % (arrive_time, name))
    
    with counter.request() as req:
        yield req 
        wait_time = env.now - arrive_time
        print('%7.4f %s: Waited %6.3f' % (env.now, name, wait_time))
        service_time = np.random.exponential(mean_service_time)
        yield env.timeout(service_time)
        print('%7.4f %s: Finished' % (env.now, name))

def source(env, customer_n, interval, counter):
    ## exponential time 마다 customer를 추가해줍니다
    for i in range(customer_n):
        c = customer(env, 'Customer%02d' % i, counter, mean_service_time=5.0)
        env.process(c)
        t = np.random.exponential(interval)
        yield env.timeout(t)
        
#np.random.seed(42)
env = simpy.Environment()

## 우선 counter generator를 만들어주고 
counter = simpy.Resource(env, capacity=1)
## 5명의 고객이, 평균적으로 3초에 한번씩 생기고, counter resource를 이용합니다. 
bank = source(env, 5, 3.0, counter)

env.process(bank)
env.run(until=90)
