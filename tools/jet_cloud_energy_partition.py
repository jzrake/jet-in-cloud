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
    f0[f0 > 1.0] = 1.0

    kinetic = dv * (d0 * h0 * u0 * (u0 - 1.0))
    thermal = dv * (p0 * (u0 - 1.0) + e0 * d0 * u0)

    return dict(
        gamma_beta=gb,
        theta=q0,
        kinetic_jet=kinetic * (0 + f0),
        kinetic_cld=kinetic * (1 - f0),
        thermal_jet=thermal * (0 + f0),
        thermal_cld=thermal * (1 - f0))



def plot_energy_partition_at_polar_angle(which, num, ax1, fname):

    db = load_checkpoint(fname)
    diag = make_diagnostic_fields(db)

    keys = ['kinetic_jet', 'kinetic_cld', 'thermal_jet', 'thermal_cld']
    results = {key: [diag[key][:,iq].sum() for iq in range(32)] for key in keys}
    theta = [diag['theta'][0,iq] for iq in range(32)]

    lw = 1.0 + 4.0 * which / num
    al = 1.0 - 0.8 * which / num
    jlabel = 'Jet' if which == 0 else None
    clabel = 'Cloud' if which == 0 else None
    ax1.plot(theta, results['kinetic_jet'], ls='-', lw=lw, c=(0.4, 0.8, 0.4), alpha=al, label=jlabel)
    ax1.plot(theta, results['kinetic_cld'], ls='-', lw=lw, c=(0.4, 0.4, 0.8), alpha=al, label=clabel)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()


    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    for which, fname in enumerate(args.filenames):
        plot_energy_partition_at_polar_angle(which, len(args.filenames), ax1, fname)

    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\theta$')

    plt.show()
