#!/usr/bin/env python3


import argparse
import os
import struct
import matplotlib.pyplot as plt
import numpy as np



def load_ndfile(filename):
    with open(filename, 'rb') as f:
        dtype = struct.unpack('8s', f.read(8))[0].decode('utf-8').strip('\x00')
        rank = struct.unpack('i', f.read(4))[0]
        dims = struct.unpack('i' * rank, f.read(4 * rank))
        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(dims)



def load_checkpoint(chkpt):
    database = dict()

    for patch in os.listdir(chkpt):

        fd = os.path.join(chkpt, patch)
        pd = dict()

        if os.path.isdir(fd):
            for field in os.listdir(fd):
                fe = os.path.join(fd, field)
                pd[field] = load_ndfile(fe)

            database[patch] = pd

    return database



def make_flattened_diagnostic_fields(db):
    ur = []
    uq = []
    d0 = []
    p0 = []
    dv = []
    q0 = []
    den_dv = []
    lar_dv = []
    tau_dv = []

    for patch in db:
        d0.append(db[patch]['primitive'][:,:,0])
        p0.append(db[patch]['primitive'][:,:,4])
        ur.append(db[patch]['primitive'][:,:,1])
        uq.append(db[patch]['primitive'][:,:,2])
        dv.append(db[patch]['cell_volume'][:,:,0])
        q0.append(db[patch]['cell_coords'][:,:,1])
        den_dv.append(db[patch]['conserved'][:,:,0])
        tau_dv.append(db[patch]['conserved'][:,:,4])
        lar_dv.append(db[patch]['conserved'][:,:,5])

    ur = np.array(ur).flatten()
    uq = np.array(uq).flatten()
    d0 = np.array(d0).flatten()
    p0 = np.array(p0).flatten()
    q0 = np.array(q0).flatten()
    dv = np.array(dv).flatten()
    den_dv = np.array(den_dv).flatten()
    tau_dv = np.array(tau_dv).flatten()
    lar_dv = np.array(lar_dv).flatten()

    u0 = (1.0 + ur * ur + uq * uq)**0.5
    e0 = p0 / d0 / (4. / 3 - 1)
    h0 = 1.0 + e0 + p0 / d0
    gb = (ur * ur + uq * uq)**0.5
    f0 = lar_dv / den_dv

    fluid_kinetic_energy = dv * (d0 * h0 * u0 * (u0 - 1.0))
    fluid_thermal_energy = dv * (p0 * (u0 - 1.0) + e0 * d0 * u0)
    fluid_kinetic_energy_jet = (0 + f0) * fluid_kinetic_energy
    fluid_kinetic_energy_cld = (1 - f0) * fluid_kinetic_energy
    fluid_thermal_energy_jet = (0 + f0) * fluid_thermal_energy
    fluid_thermal_energy_cld = (1 - f0) * fluid_thermal_energy

    return dict(
        gamma_beta=gb,
        theta=q0,
        fluid_kinetic_energy_jet=fluid_kinetic_energy_jet,
        fluid_kinetic_energy_cld=fluid_kinetic_energy_cld,
        fluid_thermal_energy_jet=fluid_thermal_energy_jet,
        fluid_thermal_energy_cld=fluid_thermal_energy_cld)



def plot_tau_gammabeta_pdf_jetcld(diag, ax, *args, **kwargs):
    gb = diag['gamma_beta']
    Etc = diag['fluid_thermal_energy_cld']
    Ekc = diag['fluid_kinetic_energy_cld']
    Etj = diag['fluid_thermal_energy_jet']
    Ekj = diag['fluid_kinetic_energy_jet']

    Ec = Etc + Ekc
    Ej = Etj + Ekj

    dEcdu, ubins = np.histogram(gb, weights=Ec, density=True, bins=np.logspace(-3, 1.0, 200))
    dEjdu, ubins = np.histogram(gb, weights=Ej, density=True, bins=np.logspace(-3, 1.0, 200))

    ax.fill_between(ubins[1:], 0.0, dEcdu * ubins[1:], alpha=0.2, step='pre', label='Cloud only')
    ax.step(ubins[1:], (dEcdu + dEjdu) * ubins[1:], label='Total')
    ax.set_xlabel(r"$\gamma \beta$")
    ax.set_ylabel(r"$dE/d(\log \gamma \beta)$")
    ax.set_xlim(1e-2, 10)
    ax.set_ylim(0, 2.25)
    ax.set_xscale('log')
    ax.legend()



def plot_tau_gammabeta_cdf_jetcld(diag, ax, *args, **kwargs):
    gb = diag['gamma_beta']
    Etc = diag['fluid_thermal_energy_cld']
    Ekc = diag['fluid_kinetic_energy_cld']
    Etj = diag['fluid_thermal_energy_jet']
    Ekj = diag['fluid_kinetic_energy_jet']

    Ec = Etc + Ekc
    Ej = Etj + Ekj
    dEcu, ubins = np.histogram(gb, weights=Ec, bins=np.logspace(-6, 1.0, 200))
    dEju, ubins = np.histogram(gb, weights=Ej, bins=np.logspace(-6, 1.0, 200))

    Ecu = np.cumsum(dEcu)
    Eju = np.cumsum(dEju)

    # Ecu /= (Ec + Ej).sum()
    # Eju /= (Ec + Ej).sum()

    ax.fill_between(ubins[1:], 0.0, Ecu, alpha=0.2, step='pre', label='Cloud only')
    ax.step(ubins[1:], Ecu + Eju, label='Total')
    ax.set_xlabel(r"$\gamma \beta$")
    ax.set_ylabel(r"$E(\gamma \beta)$")
    ax.set_xlim(1e-2, 10)
    ax.set_xscale('log')
    ax.legend()



def plot_tau_theta_pdf(diag, ax, *args, **kwargs):
    mu = np.cos(diag['theta'])
    Etc = diag['fluid_thermal_energy_cld']
    Ekc = diag['fluid_kinetic_energy_cld']
    Etj = diag['fluid_thermal_energy_jet']
    Ekj = diag['fluid_kinetic_energy_jet']

    Ec = Etc + Ekc
    Ej = Etj + Ekj

    dEcdmu, mubins = np.histogram(mu, weights=Ec, density=True, bins=np.cos(np.linspace(0.8, 0.0, 80)))
    dEjdmu, mubins = np.histogram(mu, weights=Ej, density=True, bins=np.cos(np.linspace(0.8, 0.0, 80)))

    nonzero_bins = (dEcdmu != 0) + (dEjdmu != 0)
    dEcdmu = dEcdmu[nonzero_bins]
    dEjdmu = dEjdmu[nonzero_bins]
    mubins = mubins[1:][nonzero_bins]

    ax.fill_between(np.arccos(mubins), 0.0, dEcdmu, alpha=0.2, step='pre', label='Cloud only')
    ax.step(np.arccos(mubins), dEcdmu + dEjdmu, label='Total')
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$dE/d(\cos \theta)$")
    ax.set_yscale('log')
    ax.set_ylim(0.5, 5000.0)
    ax.legend()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=['gamma-beta-cdf', 'gamma-beta-pdf', 'theta-pdf'])
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    if args.command == 'gamma-beta-cdf':
        fig = plt.figure(figsize=[6, 8])
        for n, fname in enumerate(args.filenames):
            ax = plt.subplot(len(args.filenames), 1, n + 1)
            db = load_checkpoint(fname)
            diag = make_flattened_diagnostic_fields(db)
            plot_tau_gammabeta_cdf_jetcld(diag, ax, label=fname)
            ax.xaxis.set_visible(n + 1 == len(args.filenames))
            ax.set_title(os.path.basename(fname).split('.')[0])
        fig.subplots_adjust(top=0.95, bottom=0.06, hspace=0.25)

    elif args.command == 'gamma-beta-pdf':
        fig = plt.figure(figsize=[6, 8])
        for n, fname in enumerate(args.filenames):
            ax = plt.subplot(len(args.filenames), 1, n + 1)
            db = load_checkpoint(fname)
            diag = make_flattened_diagnostic_fields(db)
            plot_tau_gammabeta_pdf_jetcld(diag, ax, label=fname)
            ax.xaxis.set_visible(n + 1 == len(args.filenames))
            ax.set_title(os.path.basename(fname).split('.')[0])
        fig.subplots_adjust(top=0.95, bottom=0.06, hspace=0.25)

    elif args.command == 'theta-pdf':
        fig = plt.figure(figsize=[6, 8])
        for n, fname in enumerate(args.filenames):
            ax = plt.subplot(len(args.filenames), 1, n + 1)
            db = load_checkpoint(fname)
            diag = make_flattened_diagnostic_fields(db)
            plot_tau_theta_pdf(diag, ax, label=fname)
            ax.xaxis.set_visible(n + 1 == len(args.filenames))
            ax.set_title(os.path.basename(fname))#.split('.')[0])
        fig.subplots_adjust(top=0.95, bottom=0.06, hspace=0.25)


    plt.show()
