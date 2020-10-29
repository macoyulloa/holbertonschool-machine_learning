Tasks
-----

#### 0\. Can I join?

By using the [Swapi API](https://intranet.hbtn.io/rltoken/PF40BRV6bWlSySBcGywYHA "Swapi API"), create a method that returns the list of ships that can hold a given number of passengers:

-   Prototype: `def availableShips(passengerCount):`
-   Don't forget the pagination
-   If no ship available, return an empty list.

#### 1\. Where I am?

By using the [Swapi API](https://intranet.hbtn.io/rltoken/PF40BRV6bWlSySBcGywYHA "Swapi API"), create a method that returns the list of names of the home planets of all `sentient` species.

-   Prototype: `def sentientPlanets():`
-   Don't forget the pagination

#### 2\. Rate me is you can!

By using the [Github API](https://intranet.hbtn.io/rltoken/VhN0vSwRSITIeGz26m9n3A "Github API"), write a script that prints the location of a specific user:

-   The user is passed as first argument of the script with the full API URL, example: `./2-user_location.py https://api.github.com/users/holbertonschool`
-   If the user doesn't exist, print `Not found`
-   If the status code is `403`, print `Reset in X min` where `X` is the number of minutes from now and the value of `X-Ratelimit-Reset`
-   Your code should not be executed when the file is imported (you should use `if __name__ == '__main__':`)

#### 3\. What will be next?

By using the [(unofficial) SpaceX API](https://intranet.hbtn.io/rltoken/Zuk0PBrNROo0CCM25pWnNA "(unofficial) SpaceX API"), write a script that displays the upcoming launch with these information:

-   Name of the launch
-   The date (in local time)
-   The rocket name
-   The name (with the locality) of the launchpad


#### 4\. How many by rocket?

By using the [(unofficial) SpaceX API](https://intranet.hbtn.io/rltoken/Zuk0PBrNROo0CCM25pWnNA "(unofficial) SpaceX API"), write a script that displays the number of launches per rocket.

-   All launches should be taking in consideration
-   Each line should contain the rocket name and the number of launches separated by `:` (format below in the example)
-   Order the result by the number launches (descending)
-   If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)
-   Your code should not be executed when the file is imported (you should use `if __name__ == '__main__':`)
