import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from astropy import constants

from lal import CreateDict
import lalsimulation as ls
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.core.utils import create_frequency_series

SOLAR_RADIUS_IN_M = constants.GM_sun.si.value / constants.c.si.value ** 2
SOLAR_RADIUS_IN_S = constants.GM_sun.si.value / constants.c.si.value ** 3
MEGA_PARSEC_SI = constants.pc.si.value * 1e6


class TF2(object):
    """
    A copy of the TaylorF2 waveform.

    Based on the implementation in
    https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiralTaylorF2.c

    This has not been rigorously tested.

    Parameters
    ----------
    mass_1: float
        Mass of the more massive object in solar masses.
    mass_2: float
        Mass of the less massive object in solar masses.
    chi_1: float
        Dimensionless aligned spin of the more massive object.
    chi_2: float
        Dimensionless aligned spin of the less massive object.
    luminosity_distance: float
        Distance to the binary in Mpc.
    lambda_1: float
        Dimensionless tidal deformability of the more massive object.
    lambda_2: float
        Dimensionless tidal deformability of the less massive object.
    """

    def __init__(
        self, mass_1, mass_2, chi_1, chi_2, luminosity_distance, lambda_1=0, lambda_2=0
    ):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.total_mass = self.mass_1 + self.mass_2
        self.symmetric_mass_ratio = self.mass_1 * self.mass_2 / np.square(self.total_mass)

        self.chi_1 = chi_1
        self.chi_2 = chi_2

        self.luminosity_distance = luminosity_distance * MEGA_PARSEC_SI

        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.param_dict = CreateDict()
        ls.SimInspiralWaveformParamsInsertTidalLambda1(self.param_dict, self.lambda_1)
        ls.SimInspiralWaveformParamsInsertTidalLambda2(self.param_dict, self.lambda_2)
        ls.SimInspiralSetQuadMonParamsFromLambdas(self.param_dict)


    def __call__(self, frequency_array, tc=0, phi_c=0):
        orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        hoff = self.amplitude(frequency_array, orbital_speed=orbital_speed) * xp.exp(
            xp.multiply(-1j, self.phase(frequency_array, 
                phi_c=phi_c, orbital_speed=orbital_speed))
        )
        return hoff

    def orbital_speed(self, frequency_array):
        orbital_speed_coefficient = np.pi * self.total_mass * SOLAR_RADIUS_IN_S
        return xp.power(xp.multiply(orbital_speed_coefficient, frequency_array), 1./3)
        #return (np.pi * self.total_mass * SOLAR_RADIUS_IN_S * frequency_array) ** (
        #    1 / 3
        #)

    def amplitude(self, frequency_array, orbital_speed=None):
        if orbital_speed is None:
            orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        amp_0 = (
            -4
            * self.mass_1
            * self.mass_2
            / self.luminosity_distance
            * SOLAR_RADIUS_IN_M
            * SOLAR_RADIUS_IN_S
            * np.sqrt(np.pi / 12)
        )
        #d_energy_d_flux = 5 / 32 / self.symmetric_mass_ratio / orbital_speed ** 9
        d_energy_d_flux_numbers = 5 / (32 * self.symmetric_mass_ratio)
        d_energy_d_flux = xp.divide(d_energy_d_flux_numbers, xp.power(orbital_speed, 9))
        #amp = amp_0 * d_energy_d_flux ** 0.5 * orbital_speed
        amp = xp.multiply(amp_0, xp.multiply(xp.sqrt(d_energy_d_flux), orbital_speed))

        return amp

    def phase(self, frequency_array, phi_c=0, orbital_speed=None):
        if orbital_speed is None:
            orbital_speed = self.orbital_speed(frequency_array=frequency_array)
        phase_coefficients = ls.SimInspiralTaylorF2AlignedPhasing(
            self.mass_1, self.mass_2, self.chi_1, self.chi_2, self.param_dict
        )
        phasing = xp.zeros_like(orbital_speed)
        #cumulative_power_frequency = orbital_speed ** -5
        cumulative_power_frequency = xp.power(orbital_speed, -5)
        log_orbital_speed = xp.log(orbital_speed)
        for ii in range(len(phase_coefficients.v)):
            #phasing += phase_coefficients.v[ii] * cumulative_power_frequency
            phasing = xp.add(phasing, xp.multiply(phase_coefficients.v[ii], 
                cumulative_power_frequency))
            #phasing += (
            #    phase_coefficients.vlogv[ii]
            #    * cumulative_power_frequency
            #    * xp.log(orbital_speed)
            #)
            cpf_los = xp.multiply(cumulative_power_frequency, log_orbital_speed)
            phasing = xp.add(phasing, xp.multiply(phase_coefficients.vlogv[ii], cpf_los))

            #cumulative_power_frequency *= orbital_speed
            cumulative_power_frequency = xp.multiply(cumulative_power_frequency, orbital_speed)

        #phasing -= 2 * phi_c + np.pi / 4
        extra_phasing_term = (2*phi_c) + (np.pi/4)
        phasing = xp.subtract(phasing, extra_phasing_term)

        return phasing

def call_cupy_tf2(
    frequency_array,
    mass_1,
    mass_2,
    chi_1,
    chi_2,
    luminosity_distance,
    theta_jn,
    phase,
    lambda_1=0,
    lambda_2=0,
    **kwargs
):

    waveform_kwargs = dict(reference_frequency=50.0, minimum_frequency=20.0)
    waveform_kwargs.update(kwargs)
    minimum_frequency = waveform_kwargs["minimum_frequency"]

    frequency_array = xp.asarray(frequency_array)

    #in_band = frequency_array >= minimum_frequency
    in_band = xp.greater_equal(frequency_array, minimum_frequency)

    

    h_out_of_band = xp.zeros(int(xp.sum(xp.bitwise_not(in_band))))

    wf = TF2(
        mass_1=mass_1,
        mass_2=mass_2,
        chi_1=chi_1,
        chi_2=chi_2,
        luminosity_distance=luminosity_distance,
        lambda_1=lambda_1,
        lambda_2=lambda_2
    )
    strain = wf(frequency_array[in_band], phi_c=phase)
    
    #h_plus = xp.hstack([h_out_of_band, strain]) * (1 + np.cos(theta_jn) ** 2) / 2
    h_plus_extra_bits = (1 + np.square(np.cos(theta_jn)))/2
    h_plus = xp.multiply(xp.hstack([h_out_of_band, strain]), h_plus_extra_bits)

    #h_cross = (
    #    xp.hstack([h_out_of_band, strain]) * xp.exp(-1j * np.pi / 2) * np.cos(theta_jn)
    #)
    h_cross_extra_bits = np.exp(-1j * np.pi / 2)*np.cos(theta_jn)
    h_cross = xp.multiply(xp.hstack([h_out_of_band, strain]), h_cross_extra_bits)


    return dict(plus=h_plus, cross=h_cross)


class TF2WFG(object):
    def __init__(
        self,
        duration,
        sampling_frequency,
        frequency_domain_source_model=call_cupy_tf2,
        waveform_arguments=None,
        parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
    ):
        if waveform_arguments is None:
            waveform_arguments = dict(minimum_frequency=10)
        self.fdsm = frequency_domain_source_model
        self.waveform_arguments = waveform_arguments
        self.frequency_array = xp.asarray(
            create_frequency_series(
                duration=duration, sampling_frequency=sampling_frequency
            )
        )
        self.conversion = parameter_conversion

    def frequency_domain_strain(self, parameters):
        parameters, _ = self.conversion(parameters.copy())
        parameters.update(self.waveform_arguments)
        return self.fdsm(self.frequency_array, **parameters)
