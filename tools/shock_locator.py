#!/usr/bin/env python3


import argparse
import os
import struct
import json
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



def load_config(chkpt):
    cfg = os.path.join(chkpt, 'config.json')
    return json.load(open(cfg))



def load_status(chkpt):
    sts = os.path.join(chkpt, 'status.json')
    return json.load(open(sts))



def make_diagnostic_fields(db):
    ur = [0] * len(db)
    uq = [0] * len(db)
    d0 = [0] * len(db)
    p0 = [0] * len(db)
    dv = [0] * len(db)
    r0 = [0] * len(db)
    q0 = [0] * len(db)
    den_dv = [0] * len(db)
    lar_dv = [0] * len(db)
    tau_dv = [0] * len(db)

    for patch in db:
        ind = int(patch.split('-')[0].split('.')[1])
        d0[ind] = db[patch]['primitive'][:,:,0]
        p0[ind] = db[patch]['primitive'][:,:,4]
        ur[ind] = db[patch]['primitive'][:,:,1]
        uq[ind] = db[patch]['primitive'][:,:,2]
        dv[ind] = db[patch]['cell_volume'][:,:,0]
        r0[ind] = db[patch]['cell_coords'][:,:,0]
        q0[ind] = db[patch]['cell_coords'][:,:,1]
        den_dv[ind] = db[patch]['conserved'][:,:,0]
        tau_dv[ind] = db[patch]['conserved'][:,:,4]
        lar_dv[ind] = db[patch]['conserved'][:,:,5]

    ur = np.array(ur)
    uq = np.array(uq)
    d0 = np.array(d0)
    p0 = np.array(p0)
    dv = np.array(dv)
    r0 = np.array(r0)
    q0 = np.array(q0)
    den_dv = np.array(den_dv)
    lar_dv = np.array(lar_dv)
    tau_dv = np.array(tau_dv)

    for f in [ur, uq, d0, p0, dv, r0, q0, den_dv, lar_dv, tau_dv]:
        f.resize(f.shape[0] * f.shape[1], f.shape[2])

    u0 = (1.0 + ur * ur + uq * uq)**0.5
    e0 = p0 / d0 / (4. / 3 - 1)
    h0 = 1.0 + e0 + p0 / d0
    gb = (ur * ur + uq * uq)**0.5
    f0 = lar_dv / den_dv
    # fluid_kinetic_energy = dv * (d0 * h0 * u0 * (u0 - 1.0))
    # fluid_thermal_energy = dv * (p0 * (u0 - 1.0) + e0 * d0 * u0)

    return dict(
        gamma_beta=gb,
        theta=q0,
        radius=r0,
        pressure=p0,
        fluid_energy = dv * (d0 * h0 * u0 * u0 - p0 - u0 * d0),
        specific_scalar=f0)



def locate_shock_index(diag, theta_index):
    j = theta_index
    p = diag['pressure']  [:,j]
    r = diag['radius']    [:,j]
    u = diag['gamma_beta'][:,j]
    shock_index = np.argmax(u)#r**4 * u * p)
    return shock_index



def shock_power_per_steradian_at_theta(diag, theta_index):
    ishock = locate_shock_index(diag, theta_index)
    j = theta_index
    p = diag['pressure']  [:,j]
    r = diag['radius']    [:,j]
    u = diag['gamma_beta'][:,j]

    shock_radius              = diag['radius'][ishock,j]
    shock_gamma_beta          = diag['gamma_beta'][ishock,j]
    shock_gamma               = (1 + shock_gamma_beta**2)**0.5
    upstream_density          = shock_radius**-3 # WARNING: assuming alpha=3 here! should read the config file
    shock_power_density       = shock_gamma_beta**3 / shock_gamma * upstream_density
    shock_power_per_steradian = shock_radius**2 * shock_power_density # TODO: account for shock normal vs. r-hat here
    return shock_power_per_steradian



def plot_shock_geometry(args):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for fname in args.filenames:
        db = load_checkpoint(fname)
        diag = make_diagnostic_fields(db)

        rshock = []
        qshock = []

        for j in range(diag['pressure'].shape[1] // 2):
            r = diag['radius'][:,j]
            shock_index = locate_shock_index(diag, j)
            rshock.append(r[shock_index])
            qshock.append(diag['theta'][0,j])

        rshock += reversed(rshock)
        qshock += reversed([-q for q in qshock])

        r = np.array(rshock)
        q = np.array(qshock)
        x = r * np.sin(q)
        y = r * np.cos(q)
        ax.plot(x, y)

    ax.set_aspect('equal')



def plot_shock_power_per_steradian(args):
    fig = plt.figure(figsize=[6, 8])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    for fname in args.filenames:
        db = load_checkpoint(fname)
        diag = make_diagnostic_fields(db)
        status = load_status(fname)

        nq = diag['radius'].shape[1] // 3
        mu = np.cos(diag['theta'][0,:nq])
        dPdmu = list(reversed([shock_power_per_steradian_at_theta(diag, j) for j in range(nq)]))

        mubins        = [0] + list(reversed(mu))
        mu            = np.cos(diag['theta'])
        E             = diag['fluid_energy']
        dEdmu, mubins = np.histogram(mu, weights=E, density=True, bins=mubins)
        nonzero_bins  = dEdmu != 0
        dEdmu         = dEdmu[nonzero_bins] * diag['fluid_energy'].sum()
        mubins        = mubins[1:][nonzero_bins]

        ax1.plot(np.arccos(mubins), dPdmu, label=os.path.basename(fname))
        ax2.step(np.arccos(mubins), dEdmu)
        # ax3.plot(np.arccos(mubins), dPdmu / dEdmu * status['time'])
        ax3.plot(np.arccos(mubins), dEdmu / dPdmu / status['time'])


    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    ax1.set_ylabel(r"$dP^{\rm sh}/d\Omega$")
    ax1.set_yscale('log')
    ax1.legend()

    ax2.set_ylabel(r"$dE/d\Omega$")
    ax2.set_yscale('log')

    ax3.set_ylabel(r"Success $dE / dP^{\rm sh} t^{-1}$")
    ax3.set_yscale('log')
    ax3.set_xlabel(r"$\theta$")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.07)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    #plot_shock_geometry(args)
    plot_shock_power_per_steradian(args)
    plt.show()
