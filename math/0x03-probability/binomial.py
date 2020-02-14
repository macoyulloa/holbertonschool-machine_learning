#!/usr/bin/env python3
" Binomial class representing a binomial Distribution "


class Binomial():
    " Binomial representation "

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        " initialized the binomial class "
        if data is None:
            if n <= 0:
                raise ValueEror("n must be a positive value")
            if p >= 0 and p <= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        " PMF for a given number of success "
        pmf = 0.5
        return pmf

    def cdf(self, k):
        " cuntinous distribution function "
        cdf = 0.5
        return cdf
