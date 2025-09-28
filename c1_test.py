from typing import Iterable
from numpy.typing import NDArray
from matplotlib.widgets import Slider

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# Physical constants
c = 2.997_924_580e8  # Speed of light [m/s]
k_B = 1.380_649e-23  # Boltzmann constant [J/K]
u = 1.660_539_068e-27  # Atomic weight unit [kg]
hbar = 6.626_070_150e-34/(2*np.pi)  # Reduced Planck's constant [Js]

# Di-nitrogen properties at ground electronic state X^1 Sigma_g^+
# Source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=1000
nu_e = 235_857  # Vibrational constant – first term [1/m] 
m = 14.003_074*u  # Weight of one Nitrogen atom [kg]
r_0 = 1.097_68e-10  # Internuclear distance [m]

# Plot constants and variables
err_plot_max = 1  # Limit of error axis (vert. axis) in units of error percent
T_max = 1_000_000_000  # Limit of temp. axis (hor. axis) in units of kelvin
c1_init = np.sqrt(np.pi)/2 * 277/2000 # Initial c_1 coefficient value
c1_max = 0.15  # Max c_1 value to choose from slider
c1_min = 0.10  # Min c_1 value to choose from slider

def burmann_series(x: NDArray, c: Iterable) -> NDArray:
    """ Returns the Bürmann series value for all inputs in `x` using 
    coefficients `c`.
    """
    expon = np.exp(-(x**2))
    sum = np.sqrt(np.pi)/2
    for k, c_k in enumerate(c, start=1):
        sum += c_k*(expon**k)

    return (2/np.sqrt(np.pi))*np.sign(x)*np.sqrt(1 - expon)*sum

def burmann_error(x: NDArray, c: Iterable) -> NDArray:
    """ Returns the absolute percentage error of the Bürmann series using 
    coefficients `c` compared to the numerical value, for all inputs in `x`.
    """
    erf_num = sp.erf(x)
    erf_est = burmann_series(x, c)
    return 100*np.abs(1 - erf_est/erf_num)

# Calculation of reduced mass (mu) and stiffness of bond (k)
mu = m*m/(m + m)
omega = 2*np.pi*c*nu_e
k = (omega**2)*mu

# Calculate characteristic temperature for vibrational energy
T_vib = hbar*omega/k_B  # [K]

# Temperature range and sqrt(a)*r_0 values for each T
T = np.linspace(T_vib, T_max, int(T_max/10_000))
beta = 1/(k_B*T)
a = beta*(1/4)*k
sqrt_a_r_0 = np.sqrt(a)*r_0

# Initial limit of error percentage as T -> oo
c1_init_limit = burmann_error(sqrt_a_r_0[-1], [c1_init])

#==============================================================================#
#-----------------------  PLOTTING CODE BELOW THIS LINE -----------------------#

# Approx. text height in percentage of vert. axis max
text_height = 0.04

# Approx. height above limit line for text in percentage of vert. axis max
text_y = 0.01

# Set up figure and axes
fig, ax = plt.subplots()
ax.set_title("Percentage error of erf($\\sqrt{a}r_0$) approximation for "
             "$\\text{N}^2$ in state $X^1 \\Sigma_g^+$\n"
             "using the Bürmann series with one and two coefficients")
ax.set_ylabel("err %        ", rotation=0)
ax.set_xlabel("T [K]")
ax.set_xlim([T_vib, T_max])
ax.set_ylim([0, err_plot_max*(1 + text_height + text_y)])
ax.set_xscale("log")


# Plot good Bürmann series with two coeff.
_ = ax.plot(T, burmann_error(sqrt_a_r_0, [31/200, -341/8000]),
            label="$c_1 = \\frac{31}{200}$, $c_2 = -\\frac{341}{8000}$",
            color="blue")

# Plot our Bürmann series with ONE coeff.
c1_graph, = ax.plot(T, burmann_error(sqrt_a_r_0, [c1_init]), 
                    label="Custom $c_1$", color="orange")

# Plot the horizontal asymptotic limit line of the ONE coeff. Bürmann series
c1_limit, = ax.plot([T_vib, T_max], [c1_init_limit, c1_init_limit], 
                    linestyle="--", 
                    color="black")

# Arrow point at start of x-axis
ax.annotate(f"$T = \\theta_{{\\text{{vib}}}} \\approx${round(T_vib)} K", 
            xy=(T_vib, 0), xytext=(7e3, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05))

# Text annotation on asymptotic limit
limit_text = ax.text(T_max/10, c1_init_limit + err_plot_max*text_y, 
                     f"Lim: {round(c1_init_limit, 3)}%")

# Make a vertically oriented slider to control c_1 coefficient
ax_c1 = fig.add_axes([0.935, 0.18, 0.0225, 0.63])
c1_slider = Slider(ax=ax_c1, label="$c_1$", orientation="vertical",
                   valmax=c1_max, valmin=c1_min, valinit=c1_init)

# The function to be called anytime the slider's value changes
# Has to take in a parameter of float (even if not used) or else it wont work
def update_graph(val: float) -> None:
    
    burmann_errors = burmann_error(sqrt_a_r_0, [c1_slider.val])
    c1_graph.set_ydata(burmann_errors)

    new_limit = burmann_error(sqrt_a_r_0[-1], [c1_slider.val])
    c1_limit.set_ydata([new_limit, new_limit])
    limit_text.set_text(f"Lim: {round(new_limit, 3)}%")

    max_err = np.max(burmann_errors)
    if max_err > err_plot_max:
        ax.set_ylim([0, max_err*(1 + text_height + text_y)])
        limit_text.set_y(new_limit + max_err*text_y)
    else:
        ax.set_ylim([0, err_plot_max*(1 + text_height + text_y)])
        limit_text.set_y(new_limit + err_plot_max*text_y)

    fig.canvas.draw_idle()

c1_slider.on_changed(update_graph)

ax.legend()
ax.grid()

plt.savefig("erf.svg")
plt.show()