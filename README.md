# Comparative Analysis Tool for Normalized Imagery and Profiles (CATNIP)
CATNIP is an image analysis tool, soon to be released as a browser-based application, in active development in the Follette Lab at Amherst College. 
It supports flexible image rendering, processing, and intensity profile construction to seamlessly produce multipanel graphics that facilitate the multiwavelength comparison and empirical exploration of circumstellar (planet-forming) disk substructures.
To learn more about its capabilities, intended science purpose, and future functionality, please refer to my [AAS 245 iPoster](https://aas245-aas.ipostersessions.com/Default.aspx?s=3F-44-24-84-D2-F5-E5-2B-D7-22-BD-BF-42-BC-FD-D2).

## iCATNIP
This is the current version of the software, which utilizes a Dash-based GUI to support large-scale image analysis and streamline the user experience.
### Getting Started
I am currently developing more advanced features, such as polynomial fitting, to enhance the tool's utility.
I plan to deploy the tool as a web application on a public platform in the next few months.
For now, the version I demonstrated at AAS 245 may be run and tested locally.

First, clone this repository and navigate to the app directory:
```
git clone https://github.com/bibihanselman/CATNIP.git
cd CATNIP/iCATNIP
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Then create and activate a new virtual environment:
```
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

Finally, run the app:
```
python app.py
```

## FIGG-CATNIP
The original version of CATNIP (called FIGG-CATNIP) was based on code written by Alexander DelFranco and Catherine Sarosi, two Follette Lab alumni.
I adapted their image processing and profiling routines to develop a sophisticated, versatile plotting backend that was operated through a Google Sheets control panel.
The notebooks cannot be run outside of the lab's Google Drive directory.
