DYNAMAP Project

This repository contains the DYNAMAP model developed in Python 3.9.16. The following diagram includes only Python files (.py), Jupyter Notebook files (.ipynb), and folders.

    └── Brazil and France
        ├── Figures First Cycle - Brazil and France
        ├── results
        │   ├── coordsP
        │   ├── coordsU
        │   ├── figures
        │   └── models
        ├── Ranking Construction - France.ipynb
        ├── Ranking Construction - Brazil.ipynb
        ├── DYNAMAP - France.ipynb
        ├── DYNAMAP - Brazil.ipynb
        ├── Coordinate Validation - France.ipynb
        └── Coordinate Validation - Brazil.ipynb
Folders:
<em><b>Figures First Cycle - Brazil and France</b></em>: Folder containing figures with the results of the voting analysis in France, Brazil, and Chile (first cycle).
<em><b>results</b></em>: Folder that stores the results of the DYNAMAP neural network. It also includes the ranking of proposals.
Files:
<em><b>Ranking Construction - France.ipynb</b></em>: This file builds a ranking of users’ political preferences from left to right for those who participated in voting in France.
<em><b>Ranking Construction - Brazil.ipynb</b></em>: This file builds a ranking of users’ political preferences from left to right for those who participated in voting in Brazil.
<em><b>DYNAMAP - France.ipynb</b></em>: Implements a neural network to obtain bidimensional coordinates of users who participated in voting in France and computes the probability of supporting options A and B in a vote, thereby representing their preferences.
<em><b>DYNAMAP - Brazil.ipynb</b></em>: Implements a neural network to obtain bidimensional coordinates of users who participated in voting in Brazil and computes the probability of supporting options A and B in a vote, thereby representing their preferences.
<em><b>Coordinate Validation - France.ipynb</b></em>: This file applies linear regression to analyze the relationship between users' coordinates in France and their reported political positions, and includes classification models to predict those positions.
<em><b>Coordinate Validation - Brazil.ipynb</b></em>: This file applies linear regression to analyze the relationship between users' coordinates in Brazil and their reported political positions, and includes classification models to predict those positions.