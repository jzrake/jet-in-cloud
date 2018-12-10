#!/usr/bin/env python3

import os
import struct
import numpy as np
import argparse



def load_ndfile(filename):
    with open(filename, 'rb') as f:
        dtype = struct.unpack('8s', f.read(8))[0].decode('utf-8').strip('\x00')
        rank = struct.unpack('i', f.read(4))[0]
        dims = struct.unpack('i' * rank, f.read(4 * rank))
        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(dims)



def load_checkpoint(btdir):
    database = dict()

    for patch in os.listdir(btdir):

        fd = os.path.join(btdir, patch)
        pd = dict()

        for field in os.listdir(fd):
            fe = os.path.join(fd, field)
            pd[field] = load_ndfile(fe)

        database[patch] = pd

    return database



def imshow_database(database):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='5%', pad=0.05)

    for key, patch in database.items():

        R = patch['vert_coords'][:,:,0]
        Q = patch['vert_coords'][:,:,1]
        D = patch['conserved'][:,:,0]
        pr = patch['conserved'][:,:,1]
        X = R * np.cos(Q)
        Y = R * np.sin(Q)

        # im1 = ax1.pcolormesh(Y, X, np.log10(D), edgecolor='none', lw=0.1)
        im1 = ax1.pcolormesh(Y, X, D, edgecolor='none', lw=0.1)#, vmin=0, vmax=1.5)
        fig.colorbar(im1, cax=cax1, orientation='vertical')

    ax1.set_title('Log density')
    ax1.set_aspect('equal')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    db = load_checkpoint(args.filenames[0])
    imshow_database(db)
