#!/usr/bin/env python3
""" location of a user """
import requests
from sys import argv


if __name__ == '__main__':
    """prints the location of a specific user
    """
    url = argv[1]
    user = requests.get(url)
    if user.status_code == 404:
        print("Not found")
    if user.status_code == 200:
        u = user.json()
        print(u['location'])
    if user.status_code == 403:
        user = user.json()
        x = 5
        print("Reset in {} min".format(x))
