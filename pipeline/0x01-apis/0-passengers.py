#!/usr/bin/env python3
""" Passangers request """
import requests


def availableShips(passengerCount):
    """  returns the list of ships that can hold a
         given number of passengers
         If no ship available, return an empty list.
    """

    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []
    while url is not None:
        page = requests.get(url)
        page = page.json()
        results = page['results']
        for ship in results:
            passengers = ship['passengers']
            passengers = passengers.replace(',', '')
            if passengers.isnumeric():
                if int(passengers) >= passengerCount:
                    ships.append(ship["name"])
        url = page["next"]
    return ships
