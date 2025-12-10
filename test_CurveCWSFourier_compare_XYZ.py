'''
Compares gamma(), gammadash(), gammadashdash() and gammadashdash() of a CurveCWSFourier and a CurveXYZFourier

Both Curves have the same dimensions and positions.

'''


import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, CurveXYZFourier


###################################################
###################################################
##################CurveCWSFourier##################
###################################################
###################################################

# CWS
s = SurfaceRZFourier.from_nphi_ntheta(150, 150, "full torus", 1)
R = s.get_rc(0, 0)
s.set_dofs([R, 1, 1])

# CWS CURVE
c_cws = CurveCWSFourier(s.mpol, s.ntor, s.x, 50, 0, s.nfp, s.stellsym)
c_cws.set_dofs([1, 0, 0, 0]) # [th_l, th_c0, phi_l, phi_c0]

###################################################
###################################################
##################CurveXYZFourier##################
###################################################
###################################################

c_xyz = CurveXYZFourier(50, 10)
c_xyz.set("xc(0)", R)
c_xyz.set("xc(1)", 1)
c_xyz.set("yc(0)", 0)
c_xyz.set("yc(1)", 0)
c_xyz.set("zs(1)", 1)

# Relative difference between the two curves using CurveXYZFourier for reference

gamma_diff =                np.abs(c_xyz.gamma() - c_cws.gamma())/                              np.abs(c_xyz.gamma() + 1e-14)
gammadash_diff =            np.abs(c_xyz.gammadash() - c_cws.gammadash()) /                     np.abs(c_xyz.gammadash() + 1e-14)
gammadashdash_diff =        np.abs(c_xyz.gammadashdash() - c_cws.gammadashdash()) /             np.abs(c_xyz.gammadashdash() + 1e-14)
gammadashdashdash_diff =    np.abs(c_xyz.gammadashdashdash() - c_cws.gammadashdashdash()) /     np.abs(c_xyz.gammadashdashdash() + 1e-14)

gamma_diff_sum = np.sum(gamma_diff)
gammadash_diff_sum = np.sum(gammadash_diff)
gammadashdash_diff_sum = np.sum(gammadashdash_diff)
gammadashdashdash_diff_sum = np.sum(gammadashdashdash_diff)

sum_diff_gammma = np.sum(gamma_diff_sum)
sum_diff_gammadash = np.sum(gammadash_diff_sum)
sum_diff_gammadashdash = np.sum(gammadashdash_diff_sum)
sum_diff_gammadashdashdash = np.sum(gammadashdashdash_diff_sum)

# bar plot of the differences
labels = [r"$\Gamma$", r"$\Gamma'$", r"$\Gamma''$", r"$\Gamma'''$"]
values = [sum_diff_gammma, sum_diff_gammadash, sum_diff_gammadashdash, sum_diff_gammadashdashdash]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x, values, width, label=r'$\sum \left|\frac{\Gamma^i_{XYZ} - \Gamma^i_{CWS}}{\Gamma^i_{XYZ}}\right|$')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r'$\sum \left|\frac{\Gamma^i_{XYZ} - \Gamma^i_{CWS}}{\Gamma^i_{XYZ}}\right|$', size=18)
# ax.set_title(r"Comparison of $\Gamma$, $\Gamma'$, $\Gamma''$ and $\Gamma'''$ between CurveCWSFourier and CurveXYZFourier")
ax.set_xticks(x)
ax.set_xticklabels(labels, size=20)
ax.tick_params(axis='y', labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(20)
# ax.legend()
fig.tight_layout()

plt.savefig("test_CurveCWSFourier_compare_XYZ.png")
plt.show()
