""" This module contains initial conditions for the Pluto-Charon system.

Data is found through JPL HORIZONS tool, with the following elements :
```txt
Target body name: Pluto (999)                     {source: plu043_merged}
Center body name: Pluto Barycenter (9)            {source: plu043_merged}
Center-site name: BODY CENTER
*******************************************************************************
Start time      : A.D. 2013-Jan-01 00:00:00.0000 TDB
Stop  time      : A.D. 2013-Jan-01 00:01:00.0000 TDB
Step-size       : 1 minutes
*******************************************************************************
Center geodetic : 0.0, 0.0, 0.0                   {E-lon(deg),Lat(deg),Alt(km)}
Center cylindric: 0.0, 0.0, 0.0                   {E-lon(deg),Dxy(km),Dz(km)}
Center radii    : (undefined)
Output units    : KM-S
Calendar mode   : Mixed Julian/Gregorian
Output type     : GEOMETRIC cartesian states
Output format   : 2 (position and velocity)
Reference frame : ICRF
```

It contains the following bodies:
- Pluto
- Charon
- Styx
- Nix
- Kerberos
- Hydra
"""

pluto = dict(
    GM =  8.696138177608748E+02,    # GM = 869.6 +/- 1.8 km^3/s^2
    x  = -1.578394145228780E+02,    vx = -1.770302559874553E-02,
    y  = -4.568272178648342E+02,    vy = -1.580136099844745E-02,
    z  = -2.071428686710970E+03,    vz =  4.836326393490980E-03,
    hash = "pluto"
)

charon = dict(
    GM =  1.058799888601881E+02,    # GM = 105.9 +/- 1.0 km^3/s^2
    x  =  1.297174384785259E+03,    vx =  1.453959508510874E-01,
    y  =  3.752602261747175E+03,    vy =  1.297771902069885E-01,
    z  =  1.701190583845352E+04,    vz = -3.972300396994122E-02,
    hash = "charon"
)

styx = dict(
    GM =  0.000000000000000E+00,
    x  = -3.057284277725837E+04,    vx =  2.328831889136845E-02,
    y  = -2.653581343448966E+04,    vy =  4.279779753969262E-02,
    z  =  1.231129089587662E+04,    vz =  1.464990283534420E-01,
    hash = "styx"
)

nix = dict(
    GM =  3.048175648169760E-03,
    x  =  9.024348780237848E+03,    vx =  1.004334400015914E-01,
    y  =  1.521073701650077E+04,    vy =  8.655248144274648E-02,
    z  =  4.559175735722126E+04,    vz = -4.794987464160145E-02,
    hash = "nix"
)

kerberos = dict(
    GM = 1.110040850536676E-03,
    x  = 2.356420702505210E+04,     vx = 7.925370256675676E-02,
    y  = 2.838003995076242E+04,     vy = 6.302200998424937E-02,
    z  = 4.457802582182780E+04,     vz = -8.170844510689033E-02,
    hash = "kerberos"
)

hydra = dict(
    GM =  3.211039206155255E-03,
    x  = -4.333132611324428E+04,    vx = -3.740010375800672E-02,
    y  = -4.362845759453865E+04,    vy = -1.849056107102880E-02,
    z  = -2.050654193573317E+04,    vz =  1.157937282701088E-01,
    hash = "hydra"
)

###############################################################################
""" Initial conditions from PLU058 instead of PLU043,
    extracted from plu058.bsp using SPICEYPY, at epoch Jan 01 2015
"""

# pluto = dict(
#     GM =  8.699633756209835e+02,
#     x  = -1.588999026346612e+02, vx = -1.774948624220436e-02,
#     y  = -4.593368793590651e+02, vy = -1.582746330800968e-02,
#     z  = -2.075860743012636e+03, vz =  4.861321722020466e-03,
#     hash = "pluto"
# )

# charon = dict(
#     GM =  1.061744232879427e+02,
#     x  =  1.302727458500751e+03, vx =  1.454330993740602e-01,
#     y  =  3.764320475283013e+03, vy =  1.296844985324072e-01,
#     z  =  1.700864153719455e+04, vz = -3.983353932627862e-02,
#     hash = "charon"
# )

# styx = dict(
#     GM =  2.000000000000000e-06,
#     x  = -3.053347795155604e+04, vx =  2.257971723544253e-02,
#     y  = -2.668959273915397e+04, vy =  4.294978829044686e-02,
#     z  =  1.212094167375058e+04, vz =  1.465931934090134e-01,
#     hash = "styx"
# )

# nix = dict(
#     GM =  1.800000000000000e-03,
#     x  =  9.025057329220292e+03, vx =  1.005691760190898e-01,
#     y  =  1.526395546680119e+04, vy =  8.641540414096996e-02,
#     z  =  4.558485807800698e+04, vz = -4.800927280884423e-02,
#     hash = "nix"
# )

# kerberos = dict(
#     GM =  9.000000000000001e-05,
#     x  =  2.357712782149651e+04, vx =  7.937703035773029e-02,
#     y  =  2.835832797555218e+04, vy =  6.295883047064497e-02,
#     z  =  4.460126109629536e+04, vz = -8.168459097413458e-02,
#     hash = "kerberos"
# )

# hydra = dict(
#     GM =  2.249146225742025e-03,
#     x  = -4.334483181407958e+04, vx = -3.741723198286544e-02,
#     y  = -4.367038373124132e+04, vy = -1.833862229691203e-02,
#     z  = -2.046553989672473e+04, vz =  1.158213529356178e-01,
#     hash = "hydra"
# )