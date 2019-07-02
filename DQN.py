import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque()
        self.batch_size = 32            # replay buffer에서 sample을 해야
        self.size_limit = 500000        # buffer의 최대 크기

    def put(self, data):
        self.buffer.append(data)
        # 오른쪽에 넣으니깐 왼쪽에서 빼라
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()

    def sample(self, n):
        return random.sample(self.buffer, n)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 마지막 단에서는 relu는 안넣음
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, gamma, optimizer, batch_size):
    # 한 episode 끝날때 마다 train 함수가 호출이 되는데 한번만 업데이트하는 것보다 10번 뽑아서 업데이트해라
    for i in range(10):
        batch = memory.sample(batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])   #dimension 맞추기위해(shape)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s, a, r, s_prime, done_mask = torch.tensor(s_lst, dtype = torch.float), \
                                      torch.tensor(a_lst), \
                                      torch.tensor(r_lst), \
                                      torch.tensor(s_prime_lst, dtype=torch.float), \
                                      torch.tensor(done_mask_lst)

        # 32개의 state shape:[32,4] batchsize가 32이고, s는 원래 4였음. => q(s)의 shape [32,2]
        # batch 처리를 하면 굉장히 빠르다.
        q_out = q(s)

        # q_out에는 각 양쪽의 q value가 있는데 그 중 하나의 action을 선택하기 위해서
        # q_a의 shape : [32,1]
        # gather(1,a)의 1은 a에서 0번째 축말고 1번째 축의 데이터를 뽑기 위해
        q_a = q_out.gather(1,a)

        # 다음 state를 q target의 input으로 넣어 나온 value들이 있다.
        # q_target shape : [32, 2]
        # max하면 [32,]
        # unsqueeze하면 [32, 1]
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        # 마지막 step에서는 다음에 reward를 받으면 안된다 그래서 done_mask를 이용
        # TD target은 s에서 a를 한 s'에서의 q를 이용하는건데 q learning은 s'에서의 max값을 구함으로 max_q_prime을 사용
        target = r + gamma * max_q_prime * done_mask

        # target과 q_a의 차이가 loss
        loss = F.smooth_l1_loss(target, q_a)

        # optimizer의 gradient를 비워준다.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')

    # q와 q_target으로 target network를 구현
    q = Qnet()
    q_target = Qnet()

    # q를 target q로 복사
    # state_dict는 model의 weight 정보를 dictionay 형태로 닮고 있음
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()

    avg_t = 0
    gamma = 0.98
    batch_size = 32

    # q만 update한다. q target은 고정되어 있다가 q의 정보를 가지고온다.
    optimizer = optim.Adam(q.parameters(), lr = 0.0005)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) # Linear annealing from 8% to 1%
        s = env.reset()

        for t in range(600):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)

            # game이 끝난 step이면 0이고 아니면 1
            # TD target 곱할때 테크
            done_mark = 0.0 if done else 1.0
            memory.put((s, a, r/200, s_prime, done_mark))
            s = s_prime

            if done:
                break
        avg_t += t

        # 메모리에 어느정도 축적이 된 상태에서 train 해야한다.
        if memory.size() > 2000:
            train(q, q_target, memory, gamma, optimizer, batch_size)

        # 20번 episode마다 q target network를 업데이트한다.
        if n_epi%20==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode : {}, Avg timestep : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                n_epi, avg_t/20.0, memory.size(), epsilon*100
            ))

            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()