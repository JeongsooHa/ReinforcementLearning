import simpy 
import numpy as np 

def Student(env, num, library, arrive_time):
    ## 학생은 랜덤 시간 이후 도착 
    yield env.timeout(arrive_time)
    print("student {} arrived library at {:6.2f}".format(num, env.now))
    waiting_time = env.now

    ## 아래와 같은 형태로 쓰면 자동으로 get, release가 된다.
    ## 단, 다른 형태로 쓸 경우에는 req = library.request(), library.release(req) 로 해주어야 함. 
    with library.request() as req:
        yield req ## resource를 사용이 가능하면 이 부분이 수행됨 
        waiting_time = env.now - waiting_time
        ## waiting_time이 0이 아닌 경우는 기다린 경우 
        if waiting_time !=0:
            print("student {} is waiting  during {:6.2f}".format(num, waiting_time))
        ## 얼마나 공부할지를 계산 
        study_time = np.random.triangular(left=5, right=10, mode=8)
        print("student {} start to  study at {:6.2f}".format(num, env.now))
        ## 학생이 공부를 시작했고 => 현재 도서관이 꽉 차 있을 경우 꽉 차있다는 것을 표현 
        if library.capacity == library.count:
            print("#### library full at  {:6.2f} ####".format(env.now))
        yield env.timeout(study_time)
        print("student {} end   to  study at {:6.2f}".format(num, env.now))
        print("#### library seat available at {:6.2f} ####".format(env.now))
        
env = simpy.Environment()
library = simpy.Resource(env, capacity=2)

for i in range(0, 5):
    arrive_time = np.random.triangular(left=1, right=8, mode=3)
    stu = Student(env, i, library, arrive_time)
    env.process(stu)

env.run(until=50)

