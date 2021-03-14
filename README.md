# KicktippPrediction

This repo is an exercise to implement a prediction for the Kicktipp game. 
The automatic tipping is based on ``https://github.com/DeastinY/KickTippTipper``.
The tipping models are base on data from ``https://www.football-data.co.uk/`` and 538's soccer predictions (``https://projects.fivethirtyeight.com/soccer-predictions/``)

There are a bunch of models and analyze scripts right now to investigate possible betting performance improvements.

In oder to use automated tipping with the model you need to follow these steps:

1. Check out ``https://github.com/MatthiasFo/KickTippTipper.git`` inside this folder. The KickTippTipper should be in the /KickTippTipper folder.
1. Create a ``credentials.json`` with the following data inside: ``{"username":  "your-login-name", "password":  "your-password"}``
1. Run the ```pick_upcoming_games.py``` script. It should automatically insert the tipps for the upcoming gameday.