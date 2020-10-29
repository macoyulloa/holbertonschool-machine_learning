#!/usr/bin/env python3
""" count number of lounches per rocket """
import requests


if __name__ == '__main__':
    """displays the number of launches per rocket.
    """
    number_launches = {}
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()
    for i in range(len(launches)):
        rocket = launches[i]['rocket']
        url_rocket = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
        rocket_name = requests.get(url_rocket).json()['name']
        if rocket_name in number_launches.keys():
            number_launches[rocket_name] += 1
        else:
            number_launches[rocket_name] = 1

    sort_orders = sorted(number_launches.items(),
                         key=lambda x: x[1], reverse=True)

    for i in sort_orders:
        print("{}: {}".format(i[0], i[1]))
