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

    return dict(
        gamma_beta=gb,
        theta=q0,
        density=d0,
        radius=r0,
        pressure=p0,
        specific_scalar=f0)



def figure1(args):
    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    for fname in args.filenames:
        db = load_checkpoint(fname)
        diag = make_diagnostic_fields(db)

        j = 0
        p = diag['pressure'][:,j]
        d = diag['density'][:,j]
        u = diag['gamma_beta'][:,j]
        f = diag['specific_scalar'][:,j]
        r = diag['radius'][:,j]
        shock_index = np.argmax(u)

        ax1.plot(r, p, label=fname)
        ax2.plot(r, d)
        ax3.plot(r, u)
        ax4.plot(r, f)
        ax1.axvline(r[shock_index], ls='--')
        ax2.axvline(r[shock_index], ls='--')
        ax3.axvline(r[shock_index], ls='--')

    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    ax4.set_xscale('log')
    # ax4.set_yscale('log')


    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(72, 85)

    ax1.set_ylabel(r'$p$')
    ax2.set_ylabel(r'$\rho$')
    ax3.set_ylabel(r'$\gamma \beta$')
    ax4.set_ylabel(r'$f$')
    ax4.set_xlabel(r"Radius (cm)")
    plt.show()



def figure2(args):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    for fname in args.filenames:
        db = load_checkpoint(fname)
        diag = make_diagnostic_fields(db)

        j = 0
        p = diag['pressure'][:,j]
        d = diag['density'][:,j]
        u = diag['gamma_beta'][:,j]
        f = diag['specific_scalar'][:,j]
        r = diag['radius'][:,j]
        shock_index = np.argmax(u)

        p[p < 0] = 1e-10
        #ax1.plot(r, d, label=r'$\rho$ ' + fname)
        #ax1.plot(r, u, label=r'$\gamma \beta$')
        ax1.plot(r, f, label=r'$f$')
        #ax1.plot(r, p, '-o', mfc='none', label=r'$p$ ' + fname)
        #ax1.plot(r, u/(abs(p/d))**0.5, '-o', mfc='none', label='Mach')

    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Radius (cm)")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()
    figure2(args)
