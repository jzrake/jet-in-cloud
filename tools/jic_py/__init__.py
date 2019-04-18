import os
import struct
import json
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

    tau = tau_dv / dv
    u0 = (1.0 + ur * ur + uq * uq)**0.5
    e0 = p0 / d0 / (4. / 3 - 1)
    h0 = 1.0 + e0 + p0 / d0
    gb = (ur * ur + uq * uq)**0.5
    vr = ur / u0
    f0 = lar_dv / den_dv
    f0[f0 > 1.0] = 1.0

    kinetic = dv * (d0 * h0 * u0 * (u0 - 1.0))
    thermal = dv * (p0 * (u0 - 1.0) + e0 * d0 * u0)
    flow_luminosity = r0**2 * vr * (tau + p0)
    shock_parameter = r0**2 * gb**2

    return dict(
        theta=q0,
        radius=r0,
        gamma_beta=gb,
        flow_luminosity=flow_luminosity,
        shock_parameter=shock_parameter,
        pressure=p0,
        density=d0,
        specific_scalar=f0,
        kinetic_jet=kinetic * (0 + f0),
        kinetic_cld=kinetic * (1 - f0),
        thermal_jet=thermal * (0 + f0),
        thermal_cld=thermal * (1 - f0))



def integrate_power_law(r0, r1, a):
    if a == 3:
        return 4 * np.pi * np.log(r1 / r0)
    else:
        return 4 * np.pi / (3 - a) * r0**a * (r1**(3 - a) - r0**(3 - a))



def get_run_dimensions(fname, echo=False):
    cfg = load_config(fname)
    a = cfg['density_index']
    tj = cfg['jet_opening_angle']
    te = cfg['jet_timescale']
    d0 = cfg['jet_density']
    u0 = cfg['jet_velocity']
    r0 = 1.0
    r1 = 100 # WARNING: setting fiducial cutoff radius to 100
    Mtot = integrate_power_law(r0, r1, a)
    Ljet = r0**2 * d0 * u0**2 * 2 * tj**2
    Ejet = Ljet * te;

    SolarMass = 2e33
    LightSpeed = 3e10
    CloudMass = 0.02 * SolarMass
    LightCrossingTime = 0.01 # WARNING: assuming inner radius is 10 light-ms
    EngineDuration = LightCrossingTime * te
    JetEnergy = Ejet / Mtot * CloudMass * LightSpeed**2
    dLdcostOnAxis = JetEnergy / EngineDuration / (2 * tj**2)

    if echo:
        print("density index                : {} (a == 3 ? {})".format(a, a == 3))
        print("opening angle                : {}".format(tj))
        print("E / M                        : {}".format(Ejet / Mtot))
        print("cloud mass (code)            : {}".format(Mtot))
        print("jet energy (code)            : {}".format(Ejet))
        print("cloud mass (g)               : {}".format(CloudMass))
        print("jet energy (cm)              : {}".format(JetEnergy))
        print("on-axis dL/dcos(t) (erg/s/Sr): {}".format(dLdcostOnAxis))

    return dict(
        SolarMass=SolarMass,
        LightSpeed=LightSpeed,
        CloudMass=CloudMass,
        LightCrossingTime=LightCrossingTime,
        EngineDuration=EngineDuration,
        JetEnergy=JetEnergy,
        dLdcostOnAxis=dLdcostOnAxis,
        dLdcostOnAxisCode=r0**2 * d0 * u0**2,
        InnerBoundaryRadius=LightCrossingTime * LightSpeed)
