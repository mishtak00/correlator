# correlator
Clone this repo by running the following in the terminal:
``
git clone https://github.com/mishtak00/correlator
``

Run the correlator on CMASS DR9 data through:
``
python driver.py mock_cmassDR9_north_3001.fits -c mock_cmassDR9_north_noREC_cut170_3001_143.0_center_catalog.fits -s -g
``
The first argument after driver.py is your galaxy data file. Argument '-c' signals the correlator that the next string read is the name of your centers data file. Currently, if -c is missing, the correlator will automatically create the centers data from centerfinder routines. Argument '-s' is for saving created functions and distributions to an automatically created 'funcs' folder. Argument '-g' is for plotting and saving graphs of distributions DD, RR, DG_RC and DC_RG in an automatically created folder 'plots'.
