#!/usr/bin/env python3
""" soonest rocket to launch """
import requests


if __name__ == '__main__':
    """prints the soonest rocket to launch
    """
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    launches = requests.get(url).json()
    date = float('inf')
    # know the upcoming launch with these information:
    for i in range(len(launches)):
        if date > launches[i]['date_unix']:
            date = launches[i]['date_unix']
            j = i
    launch_name = launches[j]['name']
    date = launches[j]['date_local']
    rocket = launches[j]['rocket']
    url_rocket = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
    rocket_name = requests.get(url_rocket).json()['name']
    launchpad = launches[j]['launchpad']
    url_launchpad = 'https://api.spacexdata.com/v4/launchpads/{}'.format(
        launchpad)
    launchpad_info = requests.get(url_launchpad).json()
    launchpad_name = launchpad_info['name']
    launchpad_locality = launchpad_info['locality']
    print("{} ({}) {} - {} ({})".format(
        launch_name, date, rocket_name, launchpad_name, launchpad_locality))
