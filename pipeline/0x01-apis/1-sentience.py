#!/usr/bin/env python3
""" Specie request """
import requests


def sentientPlanets():
    """ returns the list of names of the home planets
        of all sentient species.
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []
    while url is not None:
        page = requests.get(url)
        page = page.json()
        results = page['results']
        for specie in results:
            if (specie["classification"] == 'sentient' or
                    specie["designation"] == 'sentient'):
                if specie['homeworld'] is not None:
                    planet = requests.get(specie['homeworld'])
                    planets.append(planet.json()['name'])
        url = page["next"]
    return planets
