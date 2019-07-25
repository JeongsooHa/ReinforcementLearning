import simpy 
import numpy as np 

## 현재 env에 있는 모든 process를 쭉 접근할 수 있는 방법이 없음.
## 따라서, 매번 프로세스를 따로 리스트의 형태로 저장해주는 것이 필요함. 
current_ps = []

def clock(env, i, tick):
    ## generator에 interrupt 가 발생했을 때 종료하는 조건을 넣어주어야 함 
    while True:
        try:
            print("!!!!!",env.now)
            print(tick)
            yield env.timeout(tick)
            print("!!!!!!",env.now)
            print('clock {} ticks at {}'.format(i, env.now))
        except simpy.Interrupt:
            print("## the clock {} was interrupted".format(i))
            return None
            
def stop_any_process(env):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌. 
    while True:
        try:
            yield env.timeout(2)
            r = np.random.randint(0, len(current_ps))
            current_ps[r].interrupt()
            current_ps.remove(current_ps[r])
        except:
            print("#"*20)
            print("all process was interrupted at {}".format(env.now))
            return None
        
## environment setting
env = simpy.Environment()

## 6 개의 중간에 멈출 수 있는 clock을 만들어서 집어넣음
for i in range(0, 5):
    p = env.process(clock(env, i, 2))
    ## 새롭게 만들어진 프로세스에 대해서 외부에서 접근 방법이 없으므로, 따로 저장해두어야 함
    current_ps.append(p)

## 2초마다 process를 멈추는 generator도 넘겨줌
env.process(stop_any_process(env))

env.run(until=20)
