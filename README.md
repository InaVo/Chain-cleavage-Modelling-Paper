# Description
This repository hosts the companion jupyter notebook for the publication *Modelling and mechanistic study of polyethylene
chain cleavage during ball milling* (https://doi.org/10.26434/chemrxiv-2025-nkf6j)
It provides an executable version of the manuscript generating all figures and analyses from raw experimental data.
The raw data is hosted on the Open Science Foundation repository under DOI: https://doi.org/10.17605/OSF.IO/EKR94

# Getting started
## In Google Colab
The easiest way to run the notebook is in Google Colab.
Click the link below and run the notebook.
<a href="https://colab.research.google.com/github/InaVo/Chain-cleavage-Modelling-Paper/blob/main/Chain-Cleavage-Modelling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
At release the notebook is running python 3.12. 
Cells at the top of the notebook clone the repository, install packages and download the experimental data to the Colab runtime. Note that these steps can take a couple minutes.  

## Running the notebook locally
The notebook is optimized to run in Google Colab. When running the notebook locally delete or comment out all code cells before importing of modules. 
1. Clone the repository:
   ```sh
   git clone https://github.com/InaVo/Chain-cleavage-Modelling-Paper
   ```
2. When you have UV installed, you can install the remaining requirements by navigating to the repository and running:
   ```sh
   uv sync
   ```
3. Download the experimental data from the OSF repository.
   You can do this manually by downloading the ZIP file from [here](https://doi.org/10.17605/OSF.IO/EKR94) and unzipping the folder into the repository, or by using the [datahugger](https://github.com/J535D165/datahugger) library
4. Run the notebook in your IDE of choice. We recommend VS code as it allows for interactive plots in-line, which can be enabled using 
   ```python
   %matplotlib widget
   ```
	at the start of a cell.
	
# Citing

When utilizing code or data from this study in an academic publication please cite the following manuscript:
> Morgen, T., Mecking, S. Vollmer, I.,  Modelling and mechanistic study of polyethylene
chain cleavage during ball milling. _ChemRxiv_ (2025). https://doi.org/10.26434/chemrxiv-2025-nkf6j

Alternatively, the data itself can be cited as:

> Morgen, T., Mecking, S. Vollmer, I., Experimental data supporting: ‘On the Influence of Catalyst Pore Structure in the Catalytic Cracking of Polypropylene’. OSF http://dx.doi.org/10.17605/OSF.IO/EKR94 (2025)


# Bugs and Comments
Feels free to submit bug reports, comments or questions as issues.