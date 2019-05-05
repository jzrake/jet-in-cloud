#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import jic_py



parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--theta-index", default=0, type=int)
args = parser.parse_args()


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

db = jic_py.load_checkpoint(args.filename)
diag = jic_py.make_diagnostic_fields(db)

j = args.theta_index
u = diag['gamma_beta'][:,j]
p = diag['pressure'][:,j]
r = diag['radius'][:,j]
L = diag['flow_luminosity'][:,j]
shock_index = np.argmax(diag['shock_parameter'][:,j])

ax1.plot(r, u, label=r'$\gamma \beta_r$')
ax1.plot(r, L, label=r'$r^2 T^{0r}$')
ax1.plot(r, p, label=r'$p$')
ax1.axvline(diag['radius'][shock_index,0], lw=1, ls='--', c='k')

ax1.legend()
ax1.set_title(os.path.basename(args.filename) + ', ' + r'$\theta={:.3f}$'.format(diag['theta'][0,j]))
ax1.set_yscale('log')
ax1.set_xlabel(r"Radius (cm)")
plt.show()
