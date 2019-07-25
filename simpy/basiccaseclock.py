import simpy

## 내부에 loop와 yield가 있는 generator입니다.
## simpy에서는 generator를 기본적으로 사용하는데,
## generator는 yield를 포함한 연속된 명령 리스트? 라고 생각해도 상관없습니다.
## yield: return something and hold
def clock(env, name, tick):
    while True:
        ## clock generator가 한번 불러지면, 아래 행동을 수행
        print("{:.2f} sec: {} clock ticks".format(env.now, name))
        ## 넘겨 받는 시간만큼 멈춰둡니다.
        yield env.timeout(tick)

## Environment: simulation하려는 세계
env = simpy.Environment()

## 아래 부분에서 함수를 넘겨준다고 생각할 수 있는데, 정확히는 함수가 아니라 generator를 넘겨주는 것임.
## 0.5초마다 소리치는 클락
fast_clock = clock(env, '     fast', 0.5)
## 1초마다 소리치는 클락
slow_clock = clock(env, '     slow', 1)
very_fast_clock = clock(env, 'very fast', 0.1)

## 이 두 generator를 앞서 생성한 environment에 넘겨줍니다.
env.process(fast_clock)
env.process(slow_clock)
env.process(very_fast_clock)

## simulation 수행. 시간이 없으면 무한으로 돌기 때문에 until을 통해 끝나는 시간을 정해두는 것이 필요함.
env.run(until=2)
