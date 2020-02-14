#!/usr/bin/env python3
poly_integral = __import__('17-integrate').poly_integral
poly = [0]
print(poly)
print(poly_integral(poly, 60))
print("-" * 50)
poly = [5, 3, 0, 1]
print(poly)
print(poly_integral(poly, 100))
print("-" * 50)
poly = [5, 3, 0, 1]
print(poly)
print(poly_integral(poly, 5.2))
print("-" * 50)
poly = [5.9, 3.0, 0, 1]
print(poly)
print(poly_integral(poly))
print("-" * 50)
poly = [5]
print(poly)
print(poly_integral(poly, 500))
print("-" * 50)
poly = [5, 0, 0, 0,]
print(poly)
print(poly_integral(poly))
print("-" * 50)
poly = []
print(poly)
print(poly_integral(poly))
print("-" * 50)
