import simpy 
import numpy as np 

## 물론 이 아래 부분을 클래스로 구현을 해도 좋지만 일단은 이해를 위해서 다 함수로 표현함 
def subsubprocess(env):
    ## process의 개별 activity는 subprocess로 구성되어 있습니다. 
    print('        subsubprocess start at {:6.2f}'.format(env.now))
    for i in range(0, 2):
        print('         ',i)
        execution_time = np.random.triangular(left=1, right=2, mode=1)
        yield env.timeout(execution_time)
    print('        subsubprocess over  at {:6.2f}'.format(env.now))
def subprocess(env):
    ## process의 개별 activity는 subprocess로 구성되어 있습니다. 
    print('    subprocess start at {:6.2f}'.format(env.now))
    for i in range(0, 1):
        yield env.process(subsubprocess(env))
    print('    subprocess over  at {:6.2f}'.format(env.now))
    
def process(env, activity_lst):
    while True:
        for act in activity_lst:
            print("start {} at {:6.2f}".format(act, env.now))
            execution_time = np.random.triangular(left=3, right=10, mode=6)
            ## 모든 activity는 subprocess라고 생각한다.
            ## subprocess(env)가 종료되어야 다음 스텝으로 넘어감
            ## 즉 일종의 waiting for other process를 구현했다고 보면 됨 
            yield env.process(subprocess(env))
            ##############
            print("end   {} at {:6.2f}".format(act, env.now))
            transfer_time = np.random.triangular(left=1, right=3, mode=2)
            yield env.timeout(transfer_time)
        print('process instance ends')
        print('#'*30)
        return None
###########
env = simpy.Environment()
process1 = process(env, ["act_{}".format(i) for i in range(0, 3)])
env.process(process1)
env.run(50)
