import os
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveCWSFourier, ArclengthVariation
)

OUT_DIR = "./paper_output_cws/"
os.makedirs(OUT_DIR, exist_ok=True)

# Threshold and weight for the maximum length of each individual coil:
#LENGTH_THRESHOLD = 20
#LENGTH_WEIGHT = 1e-8

LENGTH_THRESHOLD = 20  
LENGTH_WEIGHT = 1e-8


# Threshold and weight for the coil-to-coil distance penalty in the objective function:
#CC_THRESHOLD = 0.1
#CC_WEIGHT = 100

CC_THRESHOLD = 0.1
CC_WEIGHT = 100


# Threshold and weight for the curvature penalty in the objective function:
#CURVATURE_THRESHOLD = 60
#CURVATURE_WEIGHT = 1e-5

CURVATURE_THRESHOLD = 60
CURVATURE_WEIGHT = 1e-5


# Threshold and weight for the mean squared curvature penalty in the objective function:
#MSC_THRESHOLD = 60
#MSC_WEIGHT = 1e-9
#ARCLENGTH_WEIGHT = 3e-8
#LENGTH_CON_WEIGHT = 0.1

MSC_THRESHOLD = 60
MSC_WEIGHT = 1e-9
ARCLENGTH_WEIGHT = 3e-8
LENGTH_CON_WEIGHT = 0.1

# SURFACE INPUT FILES FOR TESTING
#wout = 'input.axiTorus_nfp3_QA_final'
wout = 'input.final'

MAXITER = 2000 
ncoils = 4
order = 10 # order of dofs of cws curves
quadpoints = 300 #13 * order
ntheta = 50
nphi = 42
theta_linspace = np.linspace(0, 1, ntheta, endpoint=True)
phi_linspace = np.linspace(0, 1, nphi, endpoint=True)
# ext_via_normal_factor = 0.2565
ext_via_normal_factor = 0.1482
#0.25216216216216214


# CREATE FLUX SURFACE
s = SurfaceRZFourier.from_vmec_input(wout, range="half period", quadpoints_theta=theta_linspace, quadpoints_phi=phi_linspace)#ntheta=ntheta, nphi=nphi)
phi_linspace_full = np.linspace(0, 1, int(nphi*2*s.nfp), endpoint=True)
s_full = SurfaceRZFourier.from_vmec_input(wout, range="full torus", quadpoints_theta=theta_linspace, quadpoints_phi=phi_linspace_full)#ntheta=ntheta, nphi=int(nphi*2*s.nfp))
# CREATE COIL WINDING SURFACE SURFACE
cws = SurfaceRZFourier.from_vmec_input(wout, range="half period", quadpoints_theta=theta_linspace, quadpoints_phi=phi_linspace)#ntheta=ntheta, nphi=nphi)
cws_full = SurfaceRZFourier.from_vmec_input(wout, range="full torus", quadpoints_theta=theta_linspace, quadpoints_phi=phi_linspace_full)#ntheta=ntheta, nphi=int(nphi*2*s.nfp))

cws.extend_via_normal(ext_via_normal_factor)
cws_full.extend_via_normal(ext_via_normal_factor)

# CREATE CURVES + COILS     
base_curves = []
for i in range(ncoils):
    curve_cws = CurveCWSFourier(
        mpol=cws.mpol,
        ntor=cws.ntor,
        idofs=cws.x,
        quadpoints=quadpoints,
        order=order,
        nfp=cws.nfp,
        stellsym=cws.stellsym,
    )
    angle = (i + 0.5)*(2 * np.pi)/((2) * s.nfp * ncoils)
    curve_dofs = np.zeros(len(curve_cws.get_dofs()),)
    curve_dofs[0] = 1
    curve_dofs[2*order+2] = 0
    curve_dofs[2*order+3] = angle
    curve_cws.set_dofs(curve_dofs)
    curve_cws.fix(0)
    curve_cws.fix(2*order+2)
    base_curves.append(curve_cws)
base_currents = [Current(1)*1e5 for i in range(ncoils)]
#base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

bs = BiotSavart(coils)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
curves_to_vtk(base_curves, OUT_DIR + "base_curves_init", close=True)
pointData = {"B.n": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
cws_full.to_vtk(OUT_DIR + "cws_init")

bs.set_points(s.gamma().reshape((-1, 3)))

Jf = SquaredFlux(s, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, f="max") for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD, f="max") for i in range(len(base_curves))])
JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_ALS + J_MSC + J_LENGTH

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.3e}, Jf={jf:.3e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    print(outstr)
    return J, grad


dofs = np.copy(JF.x)

res = minimize(
    fun,
    dofs,
    jac=True,   
    method='L-BFGS-B',
    options={"maxiter": MAXITER, "maxcor": 300},
    tol=1e-15,
)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_opt", close=True)
curves_to_vtk(base_curves, OUT_DIR + "base_curves_opt", close=True)
pointData = {"B.n": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
cws_full.to_vtk(OUT_DIR + "cws_opt")
bs.set_points(s.gamma().reshape((-1, 3)))
bs.save(OUT_DIR + "biot_savart_opt.json")

J = JF.J()
jf = Jf.J()
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
outstr = f"J={J:.3e}, Jf={jf:.3e}, ⟨B·n⟩={BdotN:.1e}"
cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"

f = open(OUT_DIR + "info_file.txt", "w")
infostr1 = f"LENGTH_THRESHOLD: {LENGTH_THRESHOLD}\nLENGTH_WEIGHT: {LENGTH_WEIGHT}\nCC_THRESHOLD: {CC_THRESHOLD}\nCC_WEIGHT: {CC_WEIGHT}"
infostr2 = f"\nCURVATURE_THRESHOLD: {CURVATURE_THRESHOLD}\nCURVATURE_WEIGHT: {CURVATURE_WEIGHT}\nMSC_THRESHOLD: {MSC_THRESHOLD}\nMSC_WEIGHT: {MSC_WEIGHT}"
infostr3 = f"\nARCLENGTH_WEIGHT: {ARCLENGTH_WEIGHT}\nLENGTH_CON_WEIGHT: {LENGTH_CON_WEIGHT}"
infostr4 = f"\nMAXITER: {MAXITER}\nncoils: {ncoils}\norder: {order}\nquadpoints: {quadpoints}\nntheta: {ntheta}\nnphi: {nphi}\next_v_n: {ext_via_normal_factor}\n"
infostr = infostr1 + infostr2 + infostr3 + infostr4 + outstr
f.write(infostr)
f.close()
