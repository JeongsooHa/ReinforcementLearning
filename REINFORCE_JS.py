import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.gamma = 0.99

        # 모델 생성 input이 4차원 -> 128 -> 2차원으로 나옴
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,2)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0005)

    # layer 만들기
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 0)
        return x

    # 아이템을 주면 append하는거
    def put_data(self, item):
        self.data.append(item)

    def train(self):
        # return
        R = 0
        # Check data from behind and calculate loss
        # Run from the back to make calculating Return easier
        for r, log_prob in self.data[::-1]:
            R = r + R * self.gamma
            loss = -log_prob * R
            self.optimizer.zero_grad()
            # computelate gradient of each weight, backpropagration
            # AutoDiff dose it automatically!!
            loss.backward()
            self.optimizer.step()
        self.data = []

# 환경설정, 환경에다가 action, action에서 나온거 모아둠
def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    avg_t = 0

    for n_epi in range(10000):
        obs = env.reset()
        for t in range(600):
            obs = torch.tensor(obs, dtype=torch.float)
            out = pi(obs)
            m = Categorical(out)
            action = m.sample()
            obs, r, done, info = env.step(action.item())

            # RREINFORCE collects output in policy because episode is not complete until learning is possible.
            # log(out[action]) = log pi(s,a)
            pi.put_data((r,torch.log(out[action])))
            if done:
                break
        avg_t += t
        pi.train()
        if n_epi % 20 == 0 and n_epi != 0:
            print("# of episode : {}, Avg timestep : {}".format(n_epi, avg_t/20.0))
            avg_t = 0
    env.close()


if __name__ == '__main__':
    main()




