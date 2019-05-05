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
	d0 = cfg['jet_density']
	ur = cfg['jet_velocity']
	tj = cfg['jet_opening_angle']
	sj = cfg['jet_structure_exp']
	L1 = dim['dLdcostOnAxisCode']
	L0 = dim['dLdcostOnAxis']
	R0 = dim['InnerBoundaryRadius']

	theta_indexes = range(args.max_theta_index)
	# shock_radial_indexes = [np.argmax(diag['shock_parameter'][:,j]) for j in theta_indexes]
	shock_radial_indexes = [jic_py.locate_shock_index(diag, j) for j in theta_indexes]
	shock_radii = [R0 * diag['radius'][i,0] for i in shock_radial_indexes]
	flow_luminosities_at_shock = [L0 / L1 * diag['flow_luminosity'][shock_radial_indexes[j],j] for j in theta_indexes]
	thetas = [diag['theta'][0,j] for j in theta_indexes]
	Lengine = [L0 * np.exp(-(theta / tj)**sj) for theta in thetas]

	return dict(
		theta_indexes = theta_indexes,
		shock_radial_indexes = shock_radial_indexes,
		shock_radii = shock_radii,
		flow_luminosities_at_shock = flow_luminosities_at_shock,
		thetas = thetas,
		Lengine = Lengine)



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



parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='+')
parser.add_argument("--max-theta-index", default=64, type=int)
parser.add_argument("--show", action='store_true')
args = parser.parse_args()



if args.show:
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=[6,8])
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

colors = [(c * 0.8, c, c * 0.8) for c in np.linspace(0.6, 0.3, len(args.filenames))]

for c, fname in zip(colors, args.filenames):
	p = load_or_make_data_product(fname)

	if args.show:
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


if args.show:
	ax1.legend()
	ax2.legend()
	ax1.set_ylabel('radius (cm)')
	ax2.set_yscale('log')
	ax2.set_xlabel(r'$\theta$')
	ax2.set_ylim(1e-1 * min(flow_luminosities_at_shock), 2 * max(flow_luminosities_at_shock))
	ax2.set_ylabel(r'Luminosity (erg/s/Sr)')
	plt.show()