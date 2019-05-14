#!/usr/bin/env python3
import argparse
import pickle
import os
import numpy as np
import jic_py



def make_data_product(fname):
	db = jic_py.load_checkpoint(fname)
	diag = jic_py.make_diagnostic_fields(db)
	cfg = jic_py.load_config(fname)
	dim = jic_py.get_run_dimensions(fname)


	# Luminosity vs. theta
	tj = cfg['jet_opening_angle']
	sj = cfg['jet_structure_exp']
	L1 = dim['dLdcostOnAxisCode']
	L0 = dim['dLdcostOnAxis']
	R0 = dim['InnerBoundaryRadius']

	theta_indexes = range(args.max_theta_index)
	shock_radial_indexes = [jic_py.locate_shock_index(diag, j) for j in theta_indexes]
	shock_radii = [R0 * diag['radius'][i,0] for i in shock_radial_indexes]
	flow_luminosities_at_shock = [L0 / L1 * diag['flow_luminosity'][shock_radial_indexes[j],j] for j in theta_indexes]
	thetas = [diag['theta'][0,j] for j in theta_indexes]
	Lengine = [L0 * np.exp(-(theta / tj)**sj) for theta in thetas]


	# Energy distributions dE / d(\gamma \beta), dE / \omega, and \Gamma \beta vs. \theta
	Etc = diag['thermal_cld']
	Ekc = diag['kinetic_cld']
	Etj = diag['thermal_jet']
	Ekj = diag['kinetic_jet']
	total_energy = Etc + Ekc + Etj + Ekj

	d_energy = np.array([total_energy[:,j].sum() for j in range(0, args.max_theta_index)])
	d_omega = np.array([np.cos(diag['theta'][0,j]) - np.cos(diag['theta'][0,j + 1]) for j in range(0, args.max_theta_index)])
	d_energy_d_gamma_beta, ubins = np.histogram(diag['gamma_beta'], weights=total_energy, density=True, bins=np.logspace(-3, 1.0, 200))

	gamma_beta_of_theta = np.array([diag['gamma_beta'][:,j].max() for j in range(0, args.max_theta_index)])

	d_energy /= total_energy.sum()

	return dict(

		theta_indexes=theta_indexes,
		shock_radial_indexes=shock_radial_indexes,
		shock_radii=shock_radii,
		flow_luminosities_at_shock=flow_luminosities_at_shock,
		thetas=thetas,
		Lengine=Lengine,

		gamma_beta_bin_edges=ubins,
		d_energy_d_gamma_beta=d_energy_d_gamma_beta,
		d_energy_d_omega=d_energy / d_omega,
		gamma_beta_of_theta=gamma_beta_of_theta)



def load_or_make_data_product(fname):
	if fname.endswith('.shock_geometry'):
		print("loading result from", fname)
		return pickle.load(open(fname, 'rb'))
	else:
		pname = fname.strip('/') + '.shock_geometry'
		res = make_data_product(fname)
		with open(pname, 'wb') as outf:
			pickle.dump(res, outf)
		print("caching result to", pname)
		return res



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("filenames", nargs='+')
	parser.add_argument("--max-theta-index", default=64, type=int)
	parser.add_argument("--show", action='store_true')
	parser.add_argument("--save")
	parser.add_argument("--title")
	args = parser.parse_args()

	if args.show or args.save:
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=[6, 10])
		ax1 = fig.add_subplot(4, 1, 1)
		ax2 = fig.add_subplot(4, 1, 2)
		ax3 = fig.add_subplot(4, 1, 3)
		ax4 = fig.add_subplot(4, 1, 4)

	colors = [(c * 0.8, c, c * 0.8) for c in np.linspace(0.6, 0.3, len(args.filenames))]

	for c, fname in zip(colors, args.filenames):
		p = load_or_make_data_product(fname)

		if args.show or args.save:
			theta_indexes = p['theta_indexes']
			shock_radial_indexes = p['shock_radial_indexes']
			shock_radii = p['shock_radii']
			flow_luminosities_at_shock = p['flow_luminosities_at_shock']
			thetas = p['thetas']
			Lengine = p['Lengine']

			# label1 = r'$r_{\rm shock}$' if False and fname == args.filenames[0] else None
			label1 = r'$r_{\rm shock}$' if False and fname == args.filenames[0] else None
			label2 = r'$L(\theta, r_{\rm inner})$' if False and fname == args.filenames[0] else None

			ax1.plot(thetas, shock_radii, lw=3, c=c, label=label1)
			ax2.plot(thetas, flow_luminosities_at_shock, lw=3, c=c, label=r'$L(\theta, r_{\rm shock})$')

			if fname == args.filenames[-1] or True:
				ax2.plot(thetas, Lengine, label=label2, ls='--', lw=1, c=c)

			ax3.plot(p['thetas'], p['gamma_beta_of_theta'], c=c)
			ax4.step(p['thetas'], p['d_energy_d_omega'], c=c)


	if args.show or args.save:

		ax1.set_ylabel('radius (cm)')
		ax2.set_yscale('log')
		ax2.set_ylim(1e-1 * min(flow_luminosities_at_shock), 2 * max(flow_luminosities_at_shock))
		ax2.set_ylabel(r'Luminosity (erg/s/Sr)')
		ax2.legend()

		ax3.set_ylabel(r'$\max(\Gamma \beta)$')

		ax4.set_yscale('log')
		ax4.set_xlabel(r'$\theta$')
		ax4.set_ylabel(r'Energy Distribution (1/Sr)')

		ax1.set_xticklabels([])
		ax2.set_xticklabels([])
		ax3.set_xticklabels([])

		fig.suptitle(args.title)
		fig.subplots_adjust(top=0.92, bottom=0.06, left=0.12, right=0.96, hspace=0.1)
		if args.save:
			fig.savefig(args.save)
		else:
			plt.show()
