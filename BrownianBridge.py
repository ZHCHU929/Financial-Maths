import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress

# Initial parameters
S0 = 100     # initial stock price
K = 110     # strike price
T = 1        # time to maturity in years
H = 120      # up-and-out barrier price/value
r = 0.05     # annual risk-free rate
vol = 0.3    # volatility (%)
B = 120 
M = 1000     # number of simulations

# Function for Monte Carlo simulation
def monte_carlo_simulation(N,a):
    dt = T/N
    nudt = (r - 0.5*vol**2)*dt
    volsdt = vol*np.sqrt(dt)
    np.random.seed(a)
    sum_CT = 0


    for i in range(M):
        BARRIER = False
        St = S0

        for j in range(N):        
            epsilon = np.random.randn()
            Stn = St*np.exp(nudt + volsdt*epsilon)
            St = Stn
            if St >= H:
                BARRIER = True
                break

        if BARRIER:
            CT = 0
        else:
            CT = max(0, K - St)

        sum_CT += CT

    C0 = np.exp(-r*T)*sum_CT/M
    return C0

# Function for Brownian Bridge simulation
def brownian_bridge_simulation(N,a):
    payoff_sum = 0.0
    np.random.seed(a)
    for _ in range(M):
        S = S0
        P = S * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * np.random.randn())
        if (S < B and P < B) or (S > B and P > B):
            payoff_sum += max(K-P, 0) * (1 - np.exp(-2 * np.log(B/S) * np.log(B/P) / (vol**2 * T)))

    return np.exp(-r * T) * payoff_sum / M

# Running simulations for different N values
N_values = range(100, 500, 10)
mc_results = []
bb_results = []
differences = []
mc = []
bb = []

for N in N_values:
    for i in range (10):
        mc_value = monte_carlo_simulation(N,i)
        bb_value = brownian_bridge_simulation(N,i)
        mc_results.append(mc_value)
        bb_results.append(bb_value)
    mc_mean = np.mean(mc_results)
    bb_mean = np.mean(bb_results)
    mc.append(mc_mean)
    bb.append(bb_mean)
    differences.append(abs(mc_mean - bb_mean))

print(differences)
# Linear regression for the differences
slope, intercept, r_value, p_value, std_err = linregress(N_values, differences)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(N_values, differences, 'o-', label='Absolute Difference')
plt.plot(N_values, intercept + slope*np.array(N_values), 'r', label='Linear Regression')
plt.xlabel('Number of Steps (N)')
plt.ylabel('Absolute Difference between MC and BB')
plt.title('Difference between Monte Carlo and Brownian Bridge Simulations')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(N_values,  mc, 'o-', label='Monte Carlo')
plt.plot(N_values, bb, 'o-', label='Brownian Bridge')
plt.xlabel('Number of Steps (N)')
plt.ylabel('average payoff')
plt.title('Monte Carlo and Brownian Bridge Simulations')
plt.legend()
plt.show()
