import math
from statistics import NormalDist

def black_and_scholes(S, K, sigma, r, T, delta=0):
    d1=(math.log(S/K)+(r-delta+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    N=NormalDist(0,1).cdf
    return {
        "Call Price" : (S*math.exp(-delta*T)*N(d1)-K*math.exp(-r*T)*N(d2)),
        "Put Price"  : (K*math.exp(-r*T)*N(-d2)-S*math.exp(-delta*T)*N(-d1))
    }

if __name__=="__main__":
    print(black_and_scholes(300, 250, 0.15, 0.03, 1))