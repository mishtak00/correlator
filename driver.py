"""
Copyright (C) 2020 Gebri Mishtaku

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses.
"""

import os
import numpy as np
from argparse import ArgumentParser
from correlator import Correlator
from utils import *

# TODO: allow saving and loading of singular distributions
# TODO: link verbose arg with printout in correlator

def main():
	parser = ArgumentParser(description=" '\./'\./'\./'\./'\./'\./ Correlator \./'\./'\./'\./'\./'\./' ")
	parser.add_argument('file', metavar='GALAXY_FILE', type=str, 
		help='Galaxy catalog for the correlation.')
	parser.add_argument('-c', '--center_file', type=str, 
		help='Makes correlator load center data from given fits file. If missing, correlator will invoke \
		centerfinding voting procedure automatically.')
	parser.add_argument('-p', '--params_file', type=str, default='params.json', 
		help='If this argument is present, the cosmological parameters will be uploaded from given file instead of the default.')
	parser.add_argument('-s', '--save', action='store_true', 
		help='If this argument is present, histograms and distributions will be saved in a "funcs" folder.')
	parser.add_argument('-g', '--graphs', action='store_true', 
		help='If this argument is present, histograms and distributions will be saved in a "graphs" folder.')
	parser.add_argument('-v', '--verbose', action='store_true', 
		help='If this argument is present, the progress of the correlator will be printed out to standard output.')
	args = parser.parse_args()


	# deletes the .fits extension
	# allows for other '.'s in the args.file string
	filename = '.'.join(args.file.split('.')[:-1])

	if args.save:
		try:
			os.mkdir('funcs_{}'.format(filename))
		except FileExistsError:
			pass

	c = Correlator(args.file, center_file=args.center_file, params_file=args.params_file, 
		save=args.save, printout=args.verbose)
	# c.plot_input_data_histogram('RA')

	c.make_DD()
	c.make_randoms()
	c.make_f_theta()
	c.make_RR()
	c.make_gtr_DG_RC()
	c.make_DG_RC()
	c.make_gtr_DC_RG()
	c.make_DC_RG()
	c.make_2pcf()

	if args.graphs:
		try:
			os.mkdir('plots_{}'.format(filename))
		except FileExistsError:
			pass
		c.plot_2pcf()
		c.plot_distribution('DD')
		c.plot_distribution('RR')
		c.plot_distribution('DC_RG')
		c.plot_distribution('DG_RC')



if __name__ == '__main__':
	main()