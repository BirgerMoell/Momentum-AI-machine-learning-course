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
