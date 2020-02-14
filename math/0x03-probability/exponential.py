#!/usr/bin/env python3
" Exponential class representing a exponential Distribution "


class Exponential():
    " Exponential representation "

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        " The initialized the class "
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        " Probability Density Function "
        if x < 0:
            return 0
        pdf = self.lambtha * (Exponential.e**((-1)*(self.lambtha)*x))
        return pdf

    def cdf(self, x):
        " Cumulative distribution function "
        if x < 0:
            return 0
        return 1 - Exponential.e**((-1)*(self.lambtha)*x)
