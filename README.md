# TS_Alignment_Analysis

This repository contains the timestamp alignment test post-processing python scripts. "analyze_compiled_data.py" contains the standard implementation of the post-processing procedure. "TS_Test_tools.py" contains the library of functions called on by the standard implementation.

This repository contains all of the requirements for running the standard implementation, except for the required local Mobile Data Compiler folder structure. Mobile Data Compiler must be set up for the user in compliance with AMSP protocols for the scripts contained herein to access timestamp alignment data.

# Dependencies
numpy, scipy, pandas, matplotlib

After installing this repository, run "install_libraries.bat" to install all of the Python package dependencies.

# Pyinstaller
Once the dependencies are installed, you can run the following command from within the root directory of the repository to compile "analyze_compiled_data.py" into an executable distribution.

pyinstaller analyze_compiled_data.py
