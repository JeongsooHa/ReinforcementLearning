import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.002
gamma = 0.98

class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.loss_lst = []

        # 모델 생성 input이 4차원 -> 128 -> 2차원으로 나옴
        self.fc1 = nn.Linear(4,128)
        self.fc_pi = nn.Linear(128,2)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    # layer 만들기
    def forward(self, x):
        x = F.relu(self.fc1(x))
        pol = self.fc_pi(x)
        pi = F.softmax(pol, dim = 0)
        v = self.fc_v(x)
        return pi, v

    # loss을 주면 append하는거
    def gather_loss(self, loss):
        # unsqueeze 차원을 하나 늘려준다. => 배치 처리하려고
        self.loss_lst.append(loss.unsqueeze(0))

    def train(self):
        # loss를 모아서 배치처리한다.
        loss = torch.cat(self.loss_lst).sum()
        # loss를 평균치로 해서 학습을 해준다. loss의 scale이 다양했지만 평균을 취해서 scale를 줄인다.
        loss = loss / len(self.loss_lst)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_lst = []

# 환경설정, 환경에다가 action, action에서 나온거 모아둠
def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    avg_t = 0

    for n_epi in range(10000):
        obs = env.reset()
        for t in range(600):
            obs = torch.from_numpy(obs).float()     # numpy를 torch의 tensor로 변경
            pi, v = model(obs)                      # actor랑 critic이 모두 필요하기 때문에 확률과 value가 같이 나와야한다.
            m = Categorical(pi)
            action = m.sample()

            obs, r, done, info = env.step(action.item())
            _, next_v = model(torch.from_numpy(obs).float())

            # TD 에러
            delta = r + gamma * next_v - v

            # 앞 부분이 policy(actor)의 업데이트 부분 + 뒤에는 value(critic)의 loss 함수
            # delta.item()을 한 이유는 delta를 하면 network가 곱해진다. 그래서 backpropagation이 될때 network까지 해버린다.
            # delta를 제곱한 이유는 양과 음 모두 에러로 받기 위해서
            # 매 step에서 업데이트를 할수 있다. 하지만 여기서는 그렇게 안함 => 코딩적인 이슈
            loss =  -torch.log(pi[action]) * delta.item() + delta * delta
            # model에다가 loss를 모아줌
            model.gather_loss(loss)

            if done:
                break

        model.train()
        avg_t += t

        if n_epi % 20 == 0 and n_epi != 0:
            print("# of episode : {}, Avg timestep : {}".format(n_epi, avg_t/20.0))
            avg_t = 0
    env.close()


if __name__ == '__main__':
    main()




