# Understanding human players and Alpha Go Zero

![title](https://scontent-arn2-1.xx.fbcdn.net/v/t31.0-8/23215485_514364231064_4399164717301012176_o.jpg?oh=9b4c3020cfc79c375efcfe162e3386bb&oe=5AAA495C)

All the world's a stage,
And all the men and women merely players.

If all the worlds a stage and we are all merely players. Wouldn't it be great to simulate the game?

This week we will explore Philipp Eisen Master Thesis Simulating Human Game Play forLevel Difficulty Estimation with
Convolutional Neural Networks

http://www.diva-portal.se/smash/get/diva2:1149021/FULLTEXT01.pdf

# Abstract
This thesis presents an approach to predict the difficulty
of levels in a game by simulating game play following a policy
learned from human game play. 

Using state-action pairs tracked
from players of the game Candy Crush Saga, we train a Convolutional
Neural Network to predict an action given a game state.
The trained model then acts as a policy.

Our goal is to predict the success rate (SR) of players, from
the SR obtained by simulating game play. Previous state-ofthe-art was using Monte Carlo tree search (MCTS) or handcrafted
heuristics for game play simulation. We benchmark our
suggested approach against one using MCTS. The hypothesis is
that, using our suggested approach, predicting the players’ SR
from the SR obtained through the simulation, leads to better
estimations of the players’ SR.

Our results show that we could not only significantly improve
the predictions of the players’ SR, but also decrease the time for
game play simulation by at least 50 times.

So get excited for game night!

# Alpha Go Zero

![title](https://scontent-arn2-1.xx.fbcdn.net/v/t31.0-8/23215614_514375378724_8927753670967270156_o.jpg?oh=3aaa86c5056b6587466abb23793b52aa&oe=5AA1D49C)

# AlphaGo Zero based RL agent 
Anton Osikas implementation of the Alpha Go Zero made during during 'AI Weekend' in Stockholm


```bash
git clone https://github.com/AntonOsika/agz
```

## Structure
```python
├── README.md
├── agz.py          # MCTS logic. File can be run for visualisations etc
├── gostate.py      # Go environment
├── policyvalue.py  # Neural networks etc for evaluating board positions
├── goboard.py      # Go code 
├── scoring.py      # More go code 
└── training.py     # Training loop performing self play 
```

## Installation

```
pip install numpy
pip install keras
pip install tensorflow

python agz.py
```

## Todo
- Implement random reflections of board (mcts kind of ruins things now)
- Tune how much time is spend exploring / training (c.f. AGZ paper)
- Parallelize training and simulation.
- Use code from `agz.play_game` to create `MCTSAgent` class 
- Use same logic this on other environments
- Learn the transition dynamics of step(state, action)

(`MCTSAgent` should probably implement `.update_state` and `.decision` methods)

# Colaboratory

![title](https://i.imgur.com/dZ91fUv.png)

Colaboratory is a data analysis tool that combines text, code, and code outputs into a single collaborative document. Lets try it out together!

![title](https://i.imgur.com/jfHKE9Q.png)

https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/welcome.ipynb


