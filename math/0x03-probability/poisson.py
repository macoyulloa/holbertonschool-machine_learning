#!/usr/bin/env python3
" Poisson class representing a Poisson Distribution "


class Poisson():
    " Poisson representation "

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        " The initialized the posson class "
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        " Poisson probability density function "
        k = int(k)
        k_fact = 1
        if k <= 0:
            return 0
        for x in range(1, k+1):
            k_fact = x * k_fact
        pmf = (Poisson.e**(self.lambtha*(-1)) * (self.lambtha**k)) / k_fact
        return pmf

    def cdf(self, k):
        " Poisson Cumulative distribution function "
        list_cdf = []
        k_fact = 1
        fact = []
        k = int(k)
        if k < 0:
            return 0
        for x in range(1, k+1):
            k_fact = x * k_fact
            fact.append(k_fact)
        for i in range(len(fact)):
            pmf = (self.lambtha**i) / fact[i]
            list_cdf.append(pmf)
        print(list_cdf)
        print(fact)
        return (Poisson.e**(self.lambtha*(-1)) * sum(list_cdf))
