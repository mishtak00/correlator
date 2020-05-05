import matplotlib.pyplot as plt
plt.rc('figure', figsize=[13, 9])
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.rc('lines', lw=2.5)
plt.rc('axes', labelsize=12)
plt.rc('axes', titlesize=18)
plt.rc('axes', grid=True)
plt.rc('grid', ls='dotted')
plt.rc('xtick', labelsize=10)
plt.rc('xtick', top=True)
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=10)
plt.rc('ytick', right=True)
plt.rc('ytick.minor', visible=True)



def plot_xi(bins_s: list, xi_s: list, filename: str, ds: float, xlims: tuple):
	plt.title(f"2pcf of {filename} with $\\Delta s = {ds}$")
	plt.xlabel("$s$ $[h^{-1} Mpc]$")
	plt.ylabel(r"$\xi(s)$")
	plt.xlim(xlims)
	plt.plot(bins_s, xi_s)
	plt.savefig(f'plots_{filename}/2pcf_ds_{int(ds)}.png')
	plt.show()


def plot_dist(bins_s: list, dist: list, distname: str, filename: str, ds: float, xlims: tuple):
	plt.title(f"{distname} of {filename} with $\\Delta s = {ds}$")
	plt.xlabel("$s$ $[h^{-1} Mpc]$")
	plt.ylabel(f"{distname} $[count]$")
	plt.xlim(xlims)
	plt.plot(bins_s, dist)
	plt.savefig(f'plots_{filename}/{distname}_ds_{int(ds)}.png')
	plt.show()


def plot_data_hist(galaxy_data: list, center_data: list, filename: str):
	plt.hist(galaxy_data, label='Galaxies RA')
	plt.hist(center_data, label='Centers RA')
	plt.title('Galaxy and center counts by RA')
	plt.xlabel(r'RA $[\degree]$')
	# # plt.hist(D_G_dec, label='Galaxies DEC')
	# # plt.hist(D_C_dec, label='Centers DEC', histtype='step')
	# # plt.title('Galaxy and center counts by DEC')
	# # plt.xlabel(r'DEC $[\degree]$')
	# # plt.hist(D_G_redshift, label='Galaxies Z')
	# # plt.hist(D_C_redshift, label='Centers Z', histtype='step')
	# # plt.title('Galaxy and center counts by Z')
	# # plt.xlabel(r'Z $[h^{-1}Mpc]$')
	# # plt.hist(D_G_radii, label='Galaxies R')
	# # plt.hist(D_C_radii, label='Centers R', histtype='step')
	# # plt.title('Galaxy and center counts by R')
	# # plt.xlabel(r'R $[h^{-1}Mpc]$')
	plt.ylabel('Count')
	plt.legend()
	plt.show()
