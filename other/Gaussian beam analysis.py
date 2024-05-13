# -*- coding: utf-8 -*-
"""
Some examples to show what is possible with the simpack.

"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


# =============================================================================
# Parametric oscillation is a gaussian trap
# =============================================================================

### Symbolic expression for (generalized) gaussian beam potential
x = sym.Symbol('x', real=True)
y = sym.Symbol('y', real=True)
z = sym.Symbol('z', real=True)

z0 = sym.Symbol('z0', positive=True)
wx = sym.Symbol('wx', positive=True)
wy = sym.Symbol('wy', positive=True)
zRx = sym.Symbol('zRx', positive=True)
zRy = sym.Symbol('zRy', positive=True)

V0 = sym.Symbol('V0', real=True)

red_wx = (1 + ((z-z0) / zRx)**2)
red_wy = (1 + ((z+z0) / zRy)**2)
ex = sym.exp(-2 * x**2 / (wx**2 * red_wx))
ey = sym.exp(-2 * y**2 / (wy**2 * red_wy))

gbeam_expr = - V0 / sym.sqrt(red_wx * red_wy) * ex * ey


dx = sym.simplify(
    sym.diff(gbeam_expr, x).subs([(x, 0), (y, 0), (z, 0)]))
dy = sym.simplify(
    sym.diff(gbeam_expr, y).subs([(x, 0), (y, 0), (z, 0)]))
dz = sym.simplify(
    sym.diff(gbeam_expr, z).subs([(x, 0), (y, 0)]))
# print("dV/dx =", dx)
# print("dV/dy =", dy)
# print("dV/dz =", dz)

# print("(dV/dz)^2 =", sym.simplify(dz**2))

num_dz = (z-z0)*(zRy**2 + (z+z0)**2) + (z+z0)*(zRx**2 + (z-z0)**2)

zmin = sym.solvers.solve(num_dz, z)[0]
# print(zmin)
# sys.exit()

d2x = sym.simplify(
    sym.diff(gbeam_expr, x, x).subs([(x, 0), (y, 0)]))
d2y = sym.simplify(
    sym.diff(gbeam_expr, y, y).subs([(x, 0), (y, 0)]))
d2z = sym.simplify(
    sym.diff(gbeam_expr, z, z).subs([(x, 0), (y, 0)]))
# print("d2V/dx2 =", d2x)
# print("d2V/dy2 =", d2y)
# print("d2V/dz2 =", d2z)


gbeam_line = gbeam_expr.subs(
    [(x, 0), (y, 0), (wx, 1), (wy, 1), (V0, 1), (zRx, 2), (zRy, 3),
      (z0, 4)])
print(gbeam_line)
f = sym.lambdify(z, gbeam_line)

dz_line = dz.subs([(wx, 1), (wy, 1), (V0, 1), (zRx, 4), (zRy, 3),
  (z0, 0.5)])
ff = sym.lambdify(z, dz_line)

zz = np.linspace(-10, 10, 1001, endpoint=True)
plt.plot(zz, f(zz))












