#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_tseries(fname, ax1, ax2):

    # Iter Time JetMass JetKineticEnergy JetThermalEnergy CloudMass CloudKineticEnergy CloudThermalEnergy
    data = np.loadtxt(fname)

    t = data[:,1]
    jet_mass = data[:,2]
    jet_kine = data[:,3]
    jet_ther = data[:,4]
    cld_mass = data[:,5]
    cld_kine = data[:,6]
    cld_ther = data[:,7]

    Mcloud = cld_mass[0]

    #ax1.plot(t, cld_mass / Mcloud, label=r'$M_{\rm cloud}$')
    #ax1.plot(t, jet_mass / Mcloud, label=r'$M_{\rm jet}$')
    #ax1.plot(t, cld_kine / Mcloud, label=r'$E_{\rm kin,cloud}$')
    #ax2.plot(t, jet_kine / Mcloud, label=r'$E_{\rm kin,jet}$')
    # ax1.plot(t, jet_kine + cld_kine, label=fname)
    ax2.plot(t, cld_kine / (cld_kine + jet_kine), label='Cloud Kinetic')
    ax2.plot(t, jet_kine / (cld_kine + jet_kine), label='Jet Kinetic')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    fig = plt.figure()
    ax1 = None # fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(1, 1, 1)

    for fname in args.filenames:
        plot_tseries(fname, ax1, ax2)

    # ax1.legend()
    ax2.legend()

    plt.show()
