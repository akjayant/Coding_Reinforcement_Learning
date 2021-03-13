1) REINFORCE WITH BASELINE - Works okay with discrete action domain and continous action domains,but high variance. Leads to irrecoverable policy steps sometimes.
2) ACTOR CRITIC ONLINE VERSION - Doesn't work properly in online mode i.e, learning at each step because its not so good for optimization for neural nets. Better way is to use parallel workers for sample efficiency.
                                 
