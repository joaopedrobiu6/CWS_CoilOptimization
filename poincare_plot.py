from simsopt.field import particles_to_vtk, compute_fieldlines
from simsopt import load
import matplotlib.pyplot as plt
import time
import numpy as np
from math import ceil, sqrt
import os
from simsopt.mhd import Vmec
from mpi4py import MPI
from simsopt.util import MpiPartition

mpi = MpiPartition(8)
comm = MPI.COMM_WORLD

def pprint(*args, **kwargs): print(*args, **kwargs) if comm.rank == 0 else 1

OUT_DIR = "./poincare_plots/"
os.makedirs(OUT_DIR, exist_ok=True) if comm.rank == 0 else 1
 
ntheta = 50
nphi = 42
nzeta = 4
nfieldlines = 12
#nfieldlines = 3

tmax=3000
#tmax=200
tol=1e-15

filename_bs_final = 'biot_savart_opt.json'
coils_directory = 'output_cws_final_minimum1'

input_file = 'wout_axiTorus_nfp3_QA_final_000_000000.nc'

vmec = Vmec(input_file, ntheta=ntheta, nphi=nphi, mpi=mpi)
vmec.run()
nfp = vmec.wout.nfp

def getRZ(vmec_final, ntheta_VMEC=50, nzeta_VMEC=4, nfieldlines=4):
    nzeta = nzeta_VMEC
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    theta = np.linspace(0,2*np.pi,num=ntheta_VMEC)
    iradii = np.linspace(0,vmec_final.wout.ns-1,num=nfieldlines).round()
    iradii = [int(i) for i in iradii]
    R_final = np.zeros((nzeta, nfieldlines,ntheta_VMEC))
    Z_final = np.zeros((nzeta, nfieldlines,ntheta_VMEC))
    for itheta in range(ntheta_VMEC):
        for izeta in range(nzeta):
            for iradius in range(nfieldlines):
                for imode, xnn in enumerate(vmec_final.wout.xn):
                    angle = vmec_final.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                    R_final[izeta,iradius,itheta] += vmec_final.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                    Z_final[izeta,iradius,itheta] += vmec_final.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
    return R_final, Z_final

pprint("Obtaining VMEC final surfaces")
R, Z = getRZ(vmec, nzeta_VMEC=nzeta, nfieldlines=nfieldlines)
pprint("VMEC final surfaces obtained")


R0 = R[0,:,0]
Z0 = Z[0,:,0]

def trace_fieldlines(bfield, R0, Z0):
    t1 = time.time()
    phis = [(i/nzeta)*(2*np.pi/nfp) for i in range(nzeta)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax, tol=tol, comm=comm,
        phis=phis, stopping_criteria=[])
    t2 = time.time()
    pprint(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//12}", flush=True)
    return fieldlines_tys, fieldlines_phi_hits, phis

bs_final = load(os.path.join(coils_directory, filename_bs_final))

pprint("Starting fieldline tracing")
fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs_final, R0, Z0)

pprint('Creating Poincare plot R, Z')
if comm.rank == 0:
    r = []
    z = []
    for izeta in range(len(phis)):
        r_2D = []
        z_2D = []
        for iradius in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[iradius][-1, 1] < 0
            data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
            if data_this_phi.size == 0:
                pprint(f'No Poincare data for iradius={iradius} and izeta={izeta}')
                continue
            r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
            z_2D.append(data_this_phi[:, 4])
        r.append(r_2D)
        z.append(z_2D)
    r = np.array(r, dtype=object)
    z = np.array(z, dtype=object)

    pprint('Plotting Poincare plot')
    nrowcol = ceil(sqrt(len(phis)))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 8))
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        axs[row, col].set_title(f"$\\phi={phis[i]/np.pi:.2f}\\pi$", loc='right', y=0.0, fontsize=10)
        axs[row, col].set_xlabel("$R$", fontsize=14)
        axs[row, col].set_ylabel("$Z$", fontsize=14)
        axs[row, col].set_aspect('equal')
        axs[row, col].tick_params(direction="in")
        for j in range(nfieldlines):
            if j== 0 and i == 0:
                legend1 = 'Poincare'
                legend3 = 'CurveCWSFourier'
            else:
                legend1 = legend3 = '_nolegend_'
            axs[row, col].plot(R[i,j,:], Z[i,j,:], '-', linewidth=1.2, c='k', label = legend3)
            try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=1.3, linewidths=1.3, c='b', label = legend1)
            except Exception as e: pprint(e, i, j)

    leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'output_cws_final_new.pdf'), bbox_inches = 'tight', pad_inches = 0)