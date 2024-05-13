# -*- coding: utf-8 -*-
"""
Physical constants and default experimental parameters.
"""

import numpy as np

# =============================================================================
# ########## Physical constants ##########
# =============================================================================
from scipy.constants import physical_constants
c = physical_constants['speed of light in vacuum'][0]
h = physical_constants['Planck constant'][0]
e = physical_constants['elementary charge'][0]
alpha = physical_constants['fine-structure constant'][0]
epsilon_0 = physical_constants['vacuum electric permittivity'][0]
e = physical_constants['elementary charge'][0]
m_e = physical_constants['electron mass'][0]
mu_B = physical_constants['Bohr magneton'][0]
a_0 = physical_constants['atomic unit of length'][0]
k_B = physical_constants['Boltzmann constant'][0]
g = physical_constants['standard acceleration of gravity'][0]
u = physical_constants['atomic mass constant'][0]
Ry_c = physical_constants['Rydberg constant times c in Hz'][0]

m_Rb87 = 86.909180520*u
Ry_c_Rb87 = Ry_c/(1+m_e/m_Rb87) # Ryd cst times c with mass correction in Hz
mu_e = m_e*m_Rb87 / (m_e + m_Rb87) # reduced mass of electron + Rb
a = a_0 * (m_e/mu_e) # reduced Bohr radius
g_l = 1 - m_e/m_Rb87 # first order correction to orbital g-factor



# light-shift coefficient for F=1,2 level
LScoef_5S = -0.01816429616015425 # Hz/(W/m^2)
# light-shift coefficient for the F=1 to F'=2 repumper transistion
LScoef_5S5P = 0.023876855585657605 # Hz/(W/m^2)
beta = - LScoef_5S / LScoef_5S5P * h # J/(W/m^2)


def ponderomotive_coef(wavelength: float)-> float:
    trap_pulsation = 2*np.pi * c / wavelength # rad/s
    return alpha * h /(m_e * trap_pulsation**2)


# =============================================================================
# ########## Physical parameters ##########
# =============================================================================

########## General parameters ##########
wavelength = 0.821 * 1e-6 # m
focal_length = 16.3 * 1e-3 # m


########## Gaussian beam parameters ##########

w0 = 1.2 * 1e-6 # m

gauss_defaults = {
    'V0': 27.1*1e6 * beta, # Hz * J/Hz
    'waist': w0, # m
    'z_R': np.pi * w0**2 / wavelength, # m
    'dz': 0., # m
    }


########## BoB parameters ##########

SLM_defaults = {
    'pxsize': 20 * 1e-6, # m
    # SLM cell area horiz, vert
    'area': (0.016, 0.012), # m
    # Pupil radius
    'pupil_radius': 5.5 * 1e-3, # m
    # Laser intensity on SLM: effective waist
    'I_waist': 4. * 1e-3, # m
    # Laser intensity on SLM: effective offset
    'I_offset': 0.1, # dimensionless
    # BoB phase shift radius
    'r_bob': 3.5 * 1e-3, # m
    }




