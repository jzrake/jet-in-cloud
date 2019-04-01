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

    fluid_kinetic_energy = dv * (d0 * h0 * u0 * (u0 - 1.0))
    fluid_thermal_energy = dv * (p0 * (u0 - 1.0) + e0 * d0 * u0)
    fluid_kinetic_energy_jet = (0 + f0) * fluid_kinetic_energy
    fluid_kinetic_energy_cld = (1 - f0) * fluid_kinetic_energy
    fluid_thermal_energy_jet = (0 + f0) * fluid_thermal_energy
    fluid_thermal_energy_cld = (1 - f0) * fluid_thermal_energy


    print("max f0:", np.max(f0))


    return dict(
        gamma_beta=gb,
        theta=q0,
        fluid_kinetic_energy_jet=fluid_kinetic_energy_jet,
        fluid_kinetic_energy_cld=fluid_kinetic_energy_cld,
        fluid_thermal_energy_jet=fluid_thermal_energy_jet,
        fluid_thermal_energy_cld=fluid_thermal_energy_cld)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()


    diag = make_diagnostic_fields(load_checkpoint(args.filenames[0]))

    iq = 4
    Ej = diag['fluid_kinetic_energy_jet'][:,iq].sum()
    Ec = diag['fluid_kinetic_energy_cld'][:,iq].sum()

    print(Ej/Ec)
