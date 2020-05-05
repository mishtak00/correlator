# correlator
Clone this repo by running the following in the terminal:
```
git clone https://github.com/mishtak00/correlator
```

Run the correlator on CMASS DR9 data through:
```
python driver.py mock_cmassDR9_north_3001.fits -c mock_cmassDR9_centers.fits -p params_cmassdr9.json -s -g
```
The first argument after driver.py is your galaxy data file. 

Argument '-c' signals the correlator that the next string read is the name of your centers data file. Currently, if -c is missing, the correlator will automatically create the centers data from centerfinder routines.

Argument '-p' sets the cosmological parameters file for the calibration of correlator routines based on real distance instead of redshift.

Argument '-s' is for saving created functions and distributions to an automatically created 'funcs' folder. 

Argument '-g' is for plotting and saving graphs of distributions DD, RR, DG_RC and DC_RG in an automatically created folder 'plots'.
