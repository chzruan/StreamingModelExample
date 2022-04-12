import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline

from astropy.cosmology import Planck15
from halotools.mock_observables import tpcf_multipole

from gsm.models.skewt.from_radial_transverse import moments2skewt
from gsm.streaming_integral import real2redshift



BoxSize = 1024.0
redshift = 0.25
kms_to_Mpc = (1 + redshift) / (100 * Planck15.efunc(redshift)) # convert the velocity unit from km/s to Mpc/h



def prepare_xiR():
    r_xiR_bins = np.loadtxt(
        "rxiR_NumDen-3.0_z0.25.txt"
    )
    r_xiR_bins_centre = 0.5 * (r_xiR_bins[1:] + r_xiR_bins[:-1])
    xiR = np.loadtxt(
        "xiR_NumDen-3.0_z0.25_Box1.txt"
    )
    xiR_func = InterpolatedUnivariateSpline(r_xiR_bins_centre, xiR, ext=1)
    return xiR_func

def prepare_vlos_pdf():
    r, m10, c20, c02, c12, c30, c40, c04, c22 = np.loadtxt(
        "moments_NumDen-3.0_z0.25_Box1.csv", 
        unpack=True, 
    )
    m10 *= kms_to_Mpc

    c20 *= kms_to_Mpc**2
    c02 *= kms_to_Mpc**2

    c12 *= kms_to_Mpc**3
    c30 *= kms_to_Mpc**3

    c40 *= kms_to_Mpc**4
    c04 *= kms_to_Mpc**4
    c22 *= kms_to_Mpc**4

    vlos_pdf_skewt = moments2skewt(
        m_10=InterpolatedUnivariateSpline(r, m10, ext=1),
        c_20=InterpolatedUnivariateSpline(r, c20, ext=1),
        c_02=InterpolatedUnivariateSpline(r, c02, ext=1),
        c_12=InterpolatedUnivariateSpline(r, c12, ext=1),
        c_30=InterpolatedUnivariateSpline(r, c30, ext=1),
        c_22=InterpolatedUnivariateSpline(r, c22, ext=1),
        c_40=InterpolatedUnivariateSpline(r, c40, ext=1),
        c_04=InterpolatedUnivariateSpline(r, c04, ext=1)
    )
    return vlos_pdf_skewt

xiR_func = prepare_xiR()
vlos_pdf_skewt = prepare_vlos_pdf()

s_output_bins = np.linspace(0, 40, 40)
s_output_bins_centre = 0.5 * (s_output_bins[1:] + s_output_bins[:-1])

mu_output_bins = np.linspace(0, 1, 500)
mu_output_bins_centre = 0.5 * (mu_output_bins[1:] + mu_output_bins[:-1])

xiS_s_mu_ST = real2redshift.simps_integrate(
    s_c=s_output_bins_centre, 
    mu_c=mu_output_bins_centre, 
    twopcf_function=xiR_func,
    los_pdf_function=vlos_pdf_skewt,
)
monopole_ST = tpcf_multipole(
    xiS_s_mu_ST,
    mu_output_bins,
    order=0
)
quadrupole_ST = tpcf_multipole(
    xiS_s_mu_ST,
    mu_output_bins,
    order=2
)
hexadecapole_ST = tpcf_multipole(
    xiS_s_mu_ST,
    mu_output_bins,
    order=4
)

s_bins_sim = np.loadtxt("./sim/s_bins_sim.txt")
s_bins_centre_sim = 0.5 * (s_bins_sim[1:] + s_bins_sim[:-1])
mu_bins_sim = np.loadtxt("./sim/mu_bins_sim.txt")
mu_bins_centre_sim = 0.5 * (mu_bins_sim[1:] + mu_bins_sim[:-1])

xiS_s_mu_sim = np.loadtxt(
    "./sim/xiS_s_mu_Box1_z0.25_los2.txt"
)
monopole_sim = tpcf_multipole(
    xiS_s_mu_sim,
    mu_bins_sim,
    order=0
)
quadrupole_sim = tpcf_multipole(
    xiS_s_mu_sim,
    mu_bins_sim,
    order=2
)
hexadecapole_sim = tpcf_multipole(
    xiS_s_mu_sim,
    mu_bins_sim,
    order=4
)


fig = plt.figure()
ax0 = plt.subplot()

ax0.plot(
    s_output_bins_centre,
    s_output_bins_centre**2 * monopole_ST,
    label=r"$s^2 \xi^S_0$, streaming model"
)
ax0.plot(
    s_bins_centre_sim,
    s_bins_centre_sim**2 * monopole_sim,
    label=r"$s^2 \xi^S_0$, streaming model",
    linestyle="--"
)

ax0.plot(
    s_output_bins_centre,
    s_output_bins_centre**2 * quadrupole_ST + 100,
    label=r"$s^2 \xi^S_2 + 100$, streaming model"
)
ax0.plot(
    s_bins_centre_sim,
    s_bins_centre_sim**2 * quadrupole_sim + 100,
    label=r"$s^2 \xi^S_2 + 100$, simulation",
    linestyle="--"
)

ax0.plot(
    s_output_bins_centre,
    s_output_bins_centre**2 * hexadecapole_ST + 30,
    label=r"$s^2 \xi^S_4 + 30$, streaming model"
)
ax0.plot(
    s_bins_centre_sim,
    s_bins_centre_sim**2 * hexadecapole_sim + 30,
    label=r"$s^2 \xi^S_4 + 30$, simulation",
    linestyle="--"
)
ax0.set_xlabel(r"$s / (h^{-1}\mathrm{Mpc})$")


ax0.legend(
    ncol=2,
    frameon=False
)

plt.savefig("st_example.pdf")
