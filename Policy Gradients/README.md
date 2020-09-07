1) REINFORCE WITH BASELINE - Works okay with discrete action domain environments but is not good for continous action domains.
2) ACTOR CRITIC ONLINE VERSION - Doesn't work properly in online mode i.e, learning at each step because its not so good for optimization for neural nets
                                 Solves CartPole-v0 but struggles on LunarLander-v2 Discrete.
3) SYNCRONOUS ACTOR CRITIC VERSION - WIP
