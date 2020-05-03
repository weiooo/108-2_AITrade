# HW1

#### Define the function for European Option
```Python
import numpy as np
from scipy import stats

def EuropeanOption(S, K, T, r, sigma, Type, shares):
    
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if (Type == 'c'):
        call = S * stats.norm.cdf(d1,0.0,1.0) - K * np.exp(-r*T) * stats.norm.cdf(d2,0.0,1.0)        
        return call * shares
    
    else:
        put = K * np.exp(-r*T) * stats.norm.cdf(-1*d2,0.0,1.0) - S * stats.norm.cdf(-1*d1,0.0,1.0)
        return put * shares   

```

#### Define the function for interpolate
```Python
def interpolate(T1,r1,T2,r2,t):
    if (T2 < T1):
        T1, T2 = T2, T1
        r1, r2 = r2, r1
    
    r = r1 + (r2-r1) * (t-T1) / (T2-T1)
    return r
```

#### Define the Class for option information
```Python
class Optioninfo:
    
    def __init__(self, S, K, T, r, sigma, Type, pos, shares):
        
        self.S = S            # S     : stock price
        self.K = K            # K     : strike price
        self.T = T            # T     : time to maturity (year)
        self.r = r            # r     : interest rate
        self.sigma = sigma    # sigma : volatility of underlying asset
        self.Type = Type      # Type  : 'c' for call; 'p' for put
        self.pos = pos        # pos   : 'l' for long; 's' for short position
        self.shares = shares  # shares: shares of underlying asset
        
        
    # function for calculating delta, vega, gamma
    def delta(self):
        V0 = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma, self.Type, self.shares)
        V1 = EuropeanOption(self.S*1.01, self.K, self.T, self.r, self.sigma, self.Type, self.shares)
        delta = (V1-V0) / 0.01 * (lambda x : -1 if x == 's' else 1)(self.pos)
        self.deltavalue = delta
        
        WS_delta = delta * RW # Weighted Sensitivity
        self.WS_delta = WS_delta
        Kb_delta = (max(0, (WS_delta ** 2))) ** 0.5
        
        return Kb_delta
        
        
    def vega(self):
        change = 0.01 # change 
        sigma1 = interpolate(0.5, imvol_6m*(1+change), 1, imvol_12m, self.T)
        V0 = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma, self.Type, self.shares)
        
        # vega_6m
        V1 = EuropeanOption(self.S, self.K, self.T, self.r, sigma1, self.Type, self.shares)
        vega_6m = (V1 - V0) / (imvol_6m * change)
        WS_vega_6m = RW * vega_6m * imvol_6m
        
        # vega_12m
        sigma1 = interpolate(0.5, imvol_6m, 1, imvol_12m*(1+change), self.T)
        V1 = EuropeanOption(self.S, self.K, self.T, self.r, sigma1, self.Type, self.shares)
        vega_12m = (V1 - V0) / (imvol_12m * change)
        WS_vega_12m = RW * vega_12m * imvol_12m
        self.WSvega = [WS_vega_6m, WS_vega_12m]
        
        # intra-bucket calculation for 6m and 12m
        alpha, Tk, Tl, rho_delta = 0.01, 0.5, 1, 1
        rho = min(1, rho_delta*np.exp(-1*alpha*abs(Tk-Tl)/min(Tk,Tl)))
        Kb_vega = (max(0,WS_vega_6m**2 + WS_vega_12m**2 + 2*rho*WS_vega_6m*WS_vega_12m))**0.5
        
        return Kb_vega
        
    
    def curv(self):
        price_change = 0.35
        # when price goes up
        S1 = self.S * (1+price_change)
        V0 = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma, self.Type, self.shares)
        V1 = EuropeanOption(S1, self.K, self.T, self.r, self.sigma, self.Type, self.shares)
        CVR_up = -((V1-V0)* (lambda x : -1 if x == 's' else 1)(self.pos) - self.deltavalue*0.35)
        
        # when price goes down
        S1 = self.S * (1-price_change)
        V1 = EuropeanOption(S1, self.K, self.T, self.r, self.sigma, self.Type, self.shares)
        CVR_down = -((V1-V0)* (lambda x : -1 if x == 's' else 1)(self.pos) + self.deltavalue*0.35)
        self.CVR_up, self.CVR_down = CVR_up, CVR_down
        
        Kb = max(((max(0,CVR_up))**2)**0.5, ((max(0,CVR_down))**2)**0.5)
        Kb_curv = (max(0, Kb**2))**0.5
        
        return Kb_curv
    
    
    def totalriskcap(self):
        totalriskcap = self.delta() + self.vega() + self.curv()

        return totalriskcap
```

#### Given the value of parameter
```Python
r_6m, r_12m = 0.025, 0.028
imvol_6m, imvol_12m = 0.25, 0.30

S, K, T = 100, 100, 9/12
r_9m = interpolate(0.5, r_6m, 1, r_12m, T)
imvol_9m = interpolate(0.5, imvol_6m, 1, imvol_12m, T)

RW = 0.35 # risk weight
```

#### Question 1
```Python
asset1 = Optioninfo(S, K, T, r_9m, imvol_9m, 'c', 's', 1000)
print('%-15s%s %10.4f'%('delta',':',asset1.delta()),
      '\n%-15s%s %10.4f'%('vega',':',asset1.vega()),
      '\n%-15s%s %10.4f'%('curvature',':',asset1.curv()),
      '\n%-15s%s %10.4f'%('total risk cap',':',asset1.totalriskcap()))
```
#### Output:
```Python
delta          : 20594.1283 
vega           :  3249.7670 
curvature      : 10517.2649 
total risk cap : 34361.1602
```

#### Question 2
Two options have the same underlying asset.
```Python
asset2 = Optioninfo(S, K, T, r_9m, imvol_9m, 'p','s', 1000)
print('%-15s%s %10.4f'%('delta',':',asset2.delta()),
      '\n%-15s%s %10.4f'%('vega',':',asset2.vega()),
      '\n%-15s%s %10.4f'%('curvature',':',asset2.curv()),
      '\n%-15s%s %10.4f'%('total risk cap',':',asset2.totalriskcap()))
```

```Python
delta          : 14405.8717 
vega           :  3249.7670 
curvature      : 10517.2649 
total risk cap : 28172.9037
```

```Python
# For Portfolio
portfolio = [asset1, asset2]
rho_12 = 1 # correlation between the underlying assets of asset1 & asset2

delta_list = []
vega_list = []
cvr_up_list = []
cvr_down_list = []

for i in range(len(portfolio)):   
    delta_list.append(portfolio[i].WS_delta)
    vega_list += portfolio[i].WSvega
    cvr_up_list.append(portfolio[i].CVR_up)
    cvr_down_list.append(portfolio[i].CVR_down)

# inter-bucket calculation
# Delta
d = np.array([delta_list])
corr = np.array([[1, rho_12], [rho_12, 1]])
port_delta = (max(0, np.sum(d.T * d * corr)))**0.5

# Vega
v = np.array([vega_list])
alpha, Tk, Tl, rho_delta = 0.01, 0.5, 1, 1
rho = min(1, rho_delta*np.exp(-1*alpha*abs(Tk-Tl)/min(Tk,Tl)))
corr = np.array([[1,rho,1,rho],[rho,1,rho,1],[1,rho,1,rho],[rho,1,rho,1]])
port_vega = (max(0, np.sum(v.T * v * corr)))**0.5

# Curvature
cvr_up = np.array(cvr_up_list)
phi = (cvr_up < 0) * (np.roll(cvr_up, 1) < 0)
Kb_up = (max(0, sum(np.square(np.maximum(cvr_up, 0)) + (rho_12**2) * cvr_up * np.roll(cvr_up, 1) * (phi != True).astype(int)))) ** 0.5

cvr_down = np.array(cvr_down_list)
phi = (cvr_down < 0) * (np.roll(cvr_down, 1) < 0)
Kb_down = (max(0, sum(np.square(np.maximum(cvr_down, 0)) + (rho_12**2) * cvr_down * np.roll(cvr_down, 1) * (phi != True).astype(int)))) ** 0.5

Kb = max(Kb_up, Kb_down)
port_curv = (max(0, Kb**2))**0.5

print('%-15s%s %10.4f'%('delta',':',port_delta),
      '\n%-15s%s %10.4f'%('vega',':',port_vega),
      '\n%-15s%s %10.4f'%('curvature',':',port_curv),
      '\n%-15s%s %10.4f'%('total risk cap',':',port_delta + port_vega + port_curv))
```

#### Output:
```
delta          :  6188.2565 
vega           :  6499.5340 
curvature      : 21034.5298 
total risk cap : 33722.3204
```