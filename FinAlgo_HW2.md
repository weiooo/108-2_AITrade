# HW2

## Interest Rate Parity

```Python
import math
import numpy as np
import matplotlib.pyplot as plt

spot_6m, sp_6m = 108, -0.1221
forward = spot_6m + sp_6m

rf_6m = 0.35350 / 100
T = 0.5

# 把美元當外國貨幣; 日圓當作本國貨幣
rd = math.log(forward/spot_6m*math.exp(rf_6m*T))/T
rd
```

```Python
0.0012726097688562329
```

## Part I | Fair Value
### Simulate the FX rate by Heston model

```Python
# Parameter
S0 = 108
r = rd # jpy_6m
T = 0.5     # maturity(year)
dt = 1/365  # 1 days

# For Heston model
V0 = 0.0102401
rho = 0.128199
kappa = 1.171979
theta = 0.0141483
sigma = 0.336611
```

```Python
def HestonProb(S0, V0, r, T, dt, rho, kappa, theta, sigma, rep):
    
    n = round(T/dt) # Simulate 182 steps
    count_no = 0
    count_yes = 0
    
    for _ in range(rep):
    
        # Generate random Brownian Motion
        MU  = np.array([0, 0])
        COV = np.matrix([[1, rho], [rho, 1]])
        W   = np.random.multivariate_normal(MU, COV, n) 
        W_S = W[:,0] 
        W_V = W[:,1]

        V = [V0] + [np.nan] * n
        S = [S0] + [np.nan] * n

        for i in range(0,n):
            V[i+1] = V[i] + kappa*(theta - V[i])*dt + sigma*np.sqrt(V[i]*dt) * W_V[i]
            if (V[i+1] < 0): V[i+1] = 0

            S[i+1] = S[i] * np.exp(dt*(r-0.5*V[i]) + np.sqrt(V[i]*dt) * W_S[i])
            if (S[i+1] >= 110):
                count_no += 1
                break
            elif (S[i+1] <= 105):
                count_yes += 1
                break
            else:
                continue
    
    prob_yes = count_yes/rep
    prob_no = count_no/rep
    prob_continue = 1 - prob_yes - prob_no
    
    return prob_yes, prob_no, prob_continue 
```
### The probability of touching yes barrier

```Python
rep = 100000
np.random.seed(seed=12345)
prob_yes, prob_no, prob_continue = HestonProb(S0, V0, r, T, dt, rho, kappa, theta, sigma, rep)
prob_yes
```

```Python
0.38704
```

### Calculate Fair Value with discount rate
```Python
def HestonPrice(NP, prob_yes, r, T):
    payoff_yes = NP * (1+0.03/2)
    p = prob_yes
    fair_value = (payoff_yes*p + NP * (1-p)) * np.exp(-r*T)
    
    return fair_value
```
```Python
NP = 1000000 # 本金
r = rf_6m

C = HestonPrice(NP, prob_yes, r, T)
C
```
```Python
1004029.4087734066
```
## Part II | Greeks

### Delta

```Python
h = 0.01 * S0
Su = S0 + h
Sd = S0 - h
 
np.random.seed(seed=12345)
prob_yes, prob_no, prob_continue = HestonProb(Su, V0, r, T, dt, rho, kappa, theta, sigma, rep)
Cu = HestonPrice(NP, prob_yes, r, T)

np.random.seed(seed=12345)
prob_yes, prob_no, prob_continue = HestonProb(Sd, V0, r, T, dt, rho, kappa, theta, sigma, rep)
Cd = HestonPrice(NP, prob_yes, r, T)
```

```Python
print("Cu =", Cu,"\nCd =", Cd)
```

```Python
Cu = 1001259.6087240495 
Cd = 1006865.0922707967
```

```Python
delta = (Cu - Cd) / (2 * h) 
print("Delta=", delta)
```

```Python
Delta= -2595.1312716421885
```


### Gamma

```Python
gamma = (Cu - 2*C + Cd) / (h**2)
print("Gamma=", gamma)
```

```Python
Gamma= 56.48443761393612
```


### Vega

```Python
h = 0.0001

np.random.seed(seed=12345)
prob_yes, prob_no, prob_continue = HestonProb(S0, V0, r, T, dt, rho, kappa, theta, sigma+h, rep)
Cu = HestonPrice(NP, prob_yes, r, T)
print("Cu =", Cu,"\nC  =", C)
```

```Python
Cu = 1003960.6803582993 
C  = 1004029.4087734066
```

```Python
vega = (Cu - C) / h
print("Vega=", vega)
```

```Python
Vega= -687284.1510735452
```
