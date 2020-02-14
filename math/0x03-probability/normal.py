#!/usr/bin/env python3
" Normal class representing a normal Distribution "


class Normal():
    " Normal representation "

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        " initialized the normal class "
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sumatoria = 0
            for i in range(0, len(data)):
                x = abs((data[i] - self.mean)**2)
                sumatoria = sumatoria + x
            self.stddev = (sumatoria / len(data))**(1/2)

    def z_score(self, x):
        " calculate z-score "
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        " calcutale the x-value "
        return self.stddev * z + self.mean

    def pdf(self, x):
        " density function "
        a = 1 / ((2 * Normal.pi * (self.stddev**2))**0.5)
        b = ((x - self.mean)**2)/(2 * (self.stddev**2))
        return a * Normal.e ** ((-1)*b)

    def cdf(self, x):
        " cuntinous distribution function "
        a = (x - self.mean) / ((2**0.5) * self.stddev)
        erf = ((4/Normal.pi)**0.5)*(a-(a**3)/3+(a**5)/10-a**7/42+(a**9)/216)
        cdf = (1+erf)/2
        return cdf
