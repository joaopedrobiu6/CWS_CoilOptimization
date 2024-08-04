import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance, MeanSquaredCurvature, 
    LpCurveCurvature, CurveCWSFourier, ArclengthVariation)

def optimization_settings(LENGTH_THRESHOLD_, LENGTH_WEIGHT_, CC_THRESHOLD_, CC_WEIGHT_, CURVATURE_THRESHOLD_, 
                          CURVATURE_WEIGHT_, MSC_THRESHOLD_, MSC_WEIGHT_, ARCLENGTH_WEIGHT_, LENGTH_CON_WEIGHT_, MAXITER_):
    # Threshold and weight for the maximum length of each individual coil:
    LENGTH_THRESHOLD = LENGTH_THRESHOLD_
    LENGTH_WEIGHT = LENGTH_WEIGHT_
    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = CC_THRESHOLD_
    CC_WEIGHT = CC_WEIGHT_
    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = CURVATURE_THRESHOLD_   
    CURVATURE_WEIGHT = CURVATURE_WEIGHT_
    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = MSC_THRESHOLD_
    MSC_WEIGHT = MSC_WEIGHT_
    ARCLENGTH_WEIGHT = ARCLENGTH_WEIGHT_
    LENGTH_CON_WEIGHT = LENGTH_CON_WEIGHT_
    MAXITER = MAXITER_

    return (LENGTH_THRESHOLD, LENGTH_WEIGHT, CC_THRESHOLD, CC_WEIGHT, CURVATURE_THRESHOLD, CURVATURE_WEIGHT,
            MSC_THRESHOLD, MSC_WEIGHT, ARCLENGTH_WEIGHT, LENGTH_CON_WEIGHT, MAXITER)


(LENGTH_THRESHOLD, LENGTH_WEIGHT, CC_THRESHOLD,
 CC_WEIGHT, CURVATURE_THRESHOLD, CURVATURE_WEIGHT,
 MSC_THRESHOLD, MSC_WEIGHT, ARCLENGTH_WEIGHT, 
 LENGTH_CON_WEIGHT, MAXITER) = optimization_settings(20, 1e-8, 0.1, 100, 60, 1e-5, 20, 1e-9, 3e-8, 0.1, 50)

OUT_DIR = "./paper_evn_sweeping_2_circular/"
os.makedirs(OUT_DIR, exist_ok=True)

ncoils = 4
order = 10 # order of dofs of cws curves
quadpoints = 300 #13 * order
ntheta = 50
nphi = 42

# wout = '/home/joaobiu/simsopt_curvecws/examples/3_Advanced/input.axiTorus_nfp3_QA_final'
wout= "input.final"

# CREATE FLUX SURFACE
s = SurfaceRZFourier.from_vmec_input(wout, range="half period", ntheta=ntheta, nphi=nphi)
s_full = SurfaceRZFourier.from_vmec_input(wout, range="full torus", ntheta=ntheta, nphi=int(nphi*2*s.nfp))

def cws_and_curves(factor):

    # CREATE COIL WINDING SURFACE
    cws = SurfaceRZFourier.from_nphi_ntheta(nphi, ntheta, "half period", s.nfp)
    cws_full = SurfaceRZFourier.from_nphi_ntheta(int(nphi*2*s.nfp), ntheta, "full torus", s.nfp)

    R = s.get_rc(0, 0)
    minor_radius_factor_cws = 1 + factor/s.get_zs(1, 0)
    cws.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])
    cws_full.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])

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

    return cws, cws_full, base_curves, base_currents

#factor_values = np.arange(0.2560, 0.2570, 0.00001) #np.arange(0.250, 0.260, 0.0001)
#factor_values = np.arange(0.247, 0.261, 0.0005)
#factor_values = np.arange(0.01, 0.1, 0.002) # 0.148
# factor_values = np.arange(0.25, 0.4002, 0.002)

factor_values = np.arange(0.145, 0.155, 0.0001)


J_values = []

for i in factor_values:
    OUT_DIR2 = f"./paper_evn_sweeping_2_circular/{i:.5f}/"
    os.makedirs(OUT_DIR2, exist_ok=True)
    cws, cws_full, base_curves, base_currents = cws_and_curves(i)

    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)

    bs.set_points(s_full.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]

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
        return J, grad

    dofs = np.copy(JF.x)

    res = minimize(
        fun,
        dofs,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxcor": 300},
        tol=1e-15,
    )

    bs.set_points(s_full.gamma().reshape((-1, 3)))
    curves_to_vtk(curves, OUT_DIR2 + "curves_opt")
    curves_to_vtk(base_curves, OUT_DIR2 + "base_curves_opt")
    pointData = {"B.n": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
    s_full.to_vtk(OUT_DIR2 + "surf_opt", extra_data=pointData)
    cws_full.to_vtk(OUT_DIR2 + "cws_opt")
    bs.set_points(s.gamma().reshape((-1, 3)))
    bs.save(OUT_DIR2 + "biot_savart_opt.json")

    print(f"{i:.6f}:    {JF.J():.3e}")
    J_values.append(JF.J())

min_x = factor_values[J_values.index(min(J_values))]
min_y = min(J_values)

print(f"{min_x} {min_y}")
plt.plot(factor_values, J_values, "-o", color = "red", label = f"minimum value of J: ({min_x:.5}, {min_y:.3e})")
plt.legend()
plt.title("Extend via normal factor variation")
plt.xlabel("extend_via_normal factor")
plt.ylabel("JF.J()")

plt.savefig(f"{OUT_DIR}_opt_evn_factor.png")
plt.show()

data = np.column_stack([factor_values, J_values])
datafile_path = OUT_DIR + "data.txt"
np.savetxt(datafile_path , data, fmt=['%f','%e'])