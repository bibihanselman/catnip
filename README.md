# Comparative Analysis Tool for Normalized Imagery and Profiles (CATNIP)
The app is publicly available [here](https://catnip-interactive-947e7ff2d429.herokuapp.com/)!
![CATNIP logo.](iCATNIP/assets/newcatniplogo.png)

## Introduction
### CATNIP is a work in progress. More detailed documentation coming soon!
CATNIP is an interactive image analysis tool, soon to be released as a web application, intended for astronomical research. CATNIP is under active development in the Follette Lab at Amherst College. 
It supports flexible image rendering, processing, and intensity profile construction to facilitate the multiwavelength comparison and empirical exploration of circumstellar (planet-forming) disk substructures.
To learn more about its capabilities, intended science purpose, and future functionality, please refer to my [AAS 245 iPoster](https://aas245-aas.ipostersessions.com/Default.aspx?s=3F-44-24-84-D2-F5-E5-2B-D7-22-BD-BF-42-BC-FD-D2).

## iCATNIP
This is the current version of the software, which utilizes a Plotly Dash GUI to support large-scale image analysis and streamline the user experience.

### Running iCATNIP on your local machine

First, clone this repository and navigate to the app directory:
```
git clone https://github.com/bibihanselman/catnip.git
cd catnip/iCATNIP
```

Then create and activate a new virtual environment:
```
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Finally, run the app:
```
python interface.py
```

## FIGG-CATNIP
The original version of CATNIP (called FIGG-CATNIP) was based on code written by two Follette Lab alumni: Alexander DelFranco and Catherine Sarosi.
I adapted their image processing and profiling routines to develop a versatile plotting backend that was operated through a Google Sheets control panel.
The notebooks cannot be run outside of the lab's Google Drive directory.
