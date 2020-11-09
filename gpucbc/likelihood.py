import numpy as np

try:
    import cupy as xp
    from .cupy_utils import i0e, logsumexp
except ImportError:
    xp = np
    from scipy.special import i0e, logsumexp

from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

import lal
import lalsimulation as lalsim


class CUPYGravitationalWaveTransient(Likelihood):
    def __init__(
        self,
        interferometers,
        waveform_generator,
        priors=None,
        distance_marginalization=True,
        phase_marginalization=True,
        time_marginalization=False,
        td_antenna_pattern=False
    ):
        """

        A likelihood object, able to compute the likelihood of the data given
        some model parameters

        The simplest frequency-domain gravitational wave transient likelihood.
        Does not include time/phase marginalization.


        Parameters
        ----------
        interferometers: list
            A list of `bilby.gw.detector.Interferometer` instances - contains
            the detector data and power spectral densities
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            An object which computes the frequency-domain strain of the signal,
            given some set of parameters

        """
        Likelihood.__init__(self, dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self._noise_log_l = np.nan
        self.psds = dict()
        self.strain = dict()
        self._data_to_gpu()
        if priors is None:
            self.priors = priors
        else:
            self.priors = priors.copy()
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        if self.distance_marginalization:
            self._setup_distance_marginalization()
            priors["luminosity_distance"] = priors["luminosity_distance"].minimum
        if self.phase_marginalization:
            priors["phase"] = 0.0
        self.time_marginalization = False
        self.td_antenna_pattern = td_antenna_pattern
        if self.td_antenna_pattern:
            self._setup_gmst_interpolant()
            self.chirpTime_WFdict = lal.CreateDict()
            self.fISCO_constant = xp.asarray(np.power(1/np.sqrt(6), 3)*(1/np.pi))

    def _data_to_gpu(self):
        for ifo in self.interferometers:
            self.psds[ifo.name] = xp.asarray(
                ifo.power_spectral_density_array[ifo.frequency_mask]
            )
            self.strain[ifo.name] = xp.asarray(
                ifo.frequency_domain_strain[ifo.frequency_mask]
            )
        self.frequency_array = xp.asarray(ifo.frequency_array[ifo.frequency_mask])
        self.duration = ifo.strain_data.duration

    def _setup_gmst_interpolant(self):
        data_start_time = self.interferometers.start_time
        data_duration = self.duration

        GPStime_array = np.arange(data_start_time, 
            data_start_time + data_duration, 1)

        gmst_array = np.zeros_like(GPStime_array)
        for i in range(len(gmst_array)):
            gmst_array[i] = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(GPStime_array[i]))

        intercept, slope = np.polynomial.polynomial.polyfit(GPStime_array, gmst_array, 1, full=False)

        self.gmst_interp_intercept = xp.asarray(intercept)
        self.gmst_interp_slope = xp.asarray(slope)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(interferometers={},\n\twaveform_generator={})".format(
                self.interferometers, self.waveform_generator
            )
        )

    def noise_log_likelihood(self):
        """ Calculates the real part of noise log-likelihood

        Returns
        -------
        float: The real part of the noise log likelihood

        """
        if np.isnan(self._noise_log_l):
            log_l = 0
            for interferometer in self.interferometers:
                name = interferometer.name
                log_l -= (
                    2.0
                    / self.duration
                    * xp.sum(xp.abs(self.strain[name]) ** 2 / self.psds[name])
                )
            self._noise_log_l = float(log_l)
        return self._noise_log_l

    def log_likelihood_ratio(self):
        """ Calculates the real part of log-likelihood value

        Returns
        -------
        float: The real part of the log likelihood

        """
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(
            self.parameters
        )
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        if self.td_antenna_pattern:

            all_parameters, _ = convert_to_lal_binary_black_hole_parameters(self.parameters.copy())

            #TF2_chirpTime(self, fseries_Hz, m1, m2, chi1z, chi2z, WFdict)
            chirptimes_seconds = self.TF2_chirpTime(
                fseries_Hz=self.frequency_array, 
                m1=all_parameters["mass_1"], 
                m2=all_parameters["mass_2"], 
                chi1z=all_parameters["chi_1"], 
                chi2z=all_parameters["chi_2"], 
                WFdict=self.chirpTime_WFdict)
            timeAtFreq_GPSseconds = xp.subtract(self.parameters["geocent_time"], chirptimes_seconds)
            timeAtFreq_gmst = self.gmst_interpolant(timeAtFreq_GPSseconds)

            td_pol_tensors = self.TD_polarization_tensors(
                                    self.parameters["ra"],
                                    self.parameters["dec"],
                                    self.parameters["psi"],
                                    timeAtFreq_gmst
                                    )
        else:
            td_pol_tensors = None

        d_inner_h = 0
        h_inner_h = 0

        for interferometer in self.interferometers:
            d_inner_h_ifo, h_inner_h_ifo = self.calculate_snrs(
                interferometer=interferometer,
                waveform_polarizations=waveform_polarizations,
                TD_polarization_tensors=td_pol_tensors
            )
            d_inner_h += d_inner_h_ifo
            h_inner_h += h_inner_h_ifo

        if self.distance_marginalization:
            log_l = self.distance_marglinalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=h_inner_h
            )
        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=h_inner_h
            )
        else:
            log_l = -2 / self.duration * (h_inner_h - 2 * xp.real(d_inner_h))
        return float(log_l.real)

    def calculate_snrs(self, interferometer, waveform_polarizations, TD_polarization_tensors=None):
        name = interferometer.name
        if TD_polarization_tensors is None:
            signal_ifo = xp.sum(
                xp.vstack(
                    [
                        waveform_polarizations[mode]
                        * float(
                            interferometer.antenna_response(
                                self.parameters["ra"],
                                self.parameters["dec"],
                                self.parameters["geocent_time"],
                                self.parameters["psi"],
                                mode,
                            )
                        )
                        for mode in waveform_polarizations
                    ]
                ),
                axis=0,
            )[interferometer.frequency_mask]
        else:
            signal_ifo = xp.sum(
                xp.vstack(
                    [
                        xp.multiply(waveform_polarizations[mode],
                            self.TD_antenna_response(
                                detector_tensor=xp.asarray(interferometer.detector_tensor),
                                TD_polarization_tensor=TD_polarization_tensors[mode]))
                        for mode in waveform_polarizations
                    ]
                ),
                axis=0,
            )[interferometer.frequency_mask]


        time_delay = (
            self.parameters["geocent_time"]
            - interferometer.strain_data.start_time
            + interferometer.time_delay_from_geocenter(
                self.parameters["ra"],
                self.parameters["dec"],
                self.parameters["geocent_time"],
            )
        )

        #signal_ifo *= xp.exp(-2j * np.pi * time_delay * self.frequency_array)
        time_delay_numbers = xp.asarray(-2j * np.pi * time_delay)
        signal_ifo = xp.multiply(signal_ifo, xp.exp(xp.multiply(time_delay_numbers, self.frequency_array)))

        #d_inner_h = xp.sum(xp.conj(signal_ifo) * self.strain[name] / self.psds[name])
        d_inner_h = xp.sum(xp.divide(xp.multiply(xp.conj(signal_ifo), self.strain[name]),
            self.psds[name]))
        #h_inner_h = xp.sum(xp.abs(signal_ifo) ** 2 / self.psds[name])
        h_inner_h = xp.sum(xp.divide(xp.square(xp.abs(signal_ifo)), self.psds[name]))
        return d_inner_h, h_inner_h

    def distance_marglinalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h_array = xp.divide(d_inner_h * self.parameters["luminosity_distance"], 
                self.distance_array)
            #d_inner_h
            #* self.parameters["luminosity_distance"]
            #/ self.distance_array

        h_inner_h_array = xp.multiply(h_inner_h, 
                xp.square(xp.divide(self.parameters["luminosity_distance"], 
                    self.distance_array )))
            #h_inner_h
            #* self.parameters["luminosity_distance"] ** 2
            #/ self.distance_array ** 2
        
        if self.phase_marginalization:
            log_l_array = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h_array, h_inner_h=h_inner_h_array
            )
        else:
            log_l_array = -2 / self.duration * (
                h_inner_h_array - 2 * xp.real(d_inner_h_array)
            )
        log_l = logsumexp(log_l_array, b=self.distance_prior_array)
        return log_l

    def phase_marginalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h = xp.abs(d_inner_h)
        d_inner_h = xp.add(xp.log(i0e(d_inner_h)), d_inner_h)
        log_l = xp.multiply(-2 / self.duration, xp.subtract(h_inner_h, xp.multiply(2., d_inner_h)))
        #log_l = -2 / self.duration * (h_inner_h - 2 * d_inner_h)
        return log_l

    def _setup_distance_marginalization(self):
        self.distance_array = np.linspace(
            self.priors["luminosity_distance"].minimum,
            self.priors["luminosity_distance"].maximum,
            10000,
        )
        self.distance_prior_array = xp.asarray(
            self.priors["luminosity_distance"].prob(self.distance_array)
        ) * (self.distance_array[1] - self.distance_array[0])
        self.distance_array = xp.asarray(self.distance_array)

    def generate_posterior_sample_from_marginalized_likelihood(self):
        return self.parameters.copy()

    def gmst_interpolant(self, GPStimes):
        return xp.add(self.gmst_interp_intercept, xp.multiply(GPStimes, self.gmst_interp_slope))

    def TD_antenna_response(self, detector_tensor, TD_polarization_tensor):
        return xp.einsum('ij,ij...->...', detector_tensor, TD_polarization_tensor)

    def TD_ra_dec_to_theta_phi(self, ra, dec, gmst_array):
    
        phi = xp.subtract(ra, gmst_array)
        theta_value = (np.pi/2) - dec # switch to internal pi
        theta = xp.full_like(phi, theta_value)

        return theta, phi

    def TD_polarization_tensors(self, ra, dec, psi, gmst_array):

        theta, phi = self.TD_ra_dec_to_theta_phi(ra, dec, gmst_array)
    
        sin_theta = xp.sin(theta)
        cos_theta = xp.cos(theta)
    
        sin_phi = xp.sin(phi)
        cos_phi = xp.cos(phi)
    
    
        u = xp.array([xp.multiply(cos_phi, cos_theta), xp.multiply(sin_phi, cos_theta), \
                  xp.multiply(-1., sin_theta)])
        v = xp.array([xp.multiply(-1., sin_phi), cos_phi, xp.zeros_like(gmst_array)])
    
        sin_psi = xp.sin(psi)
        cos_psi = xp.cos(psi)
    
        minus_u = xp.multiply(-1, u)
    
        m = xp.subtract(xp.multiply(minus_u, sin_psi), xp.multiply(v, cos_psi))
        n = xp.add(xp.multiply(minus_u, cos_psi), xp.multiply(v, sin_psi))
    
        plus_pol_tensor = xp.einsum('i...,j...->ij...', m, m) - xp.einsum('i...,j...->ij...', n, n)
        cross_pol_tensor = xp.einsum('i...,j...->ij...', m, n) + xp.einsum('i...,j...->ij...', n, m)
    
        return dict(plus = plus_pol_tensor, cross = cross_pol_tensor)

    def TF2_PhaseDerivative(self, Mf, pn_coeff):
        # see https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomD_internals.c#L1102
        
        v = xp.cbrt(xp.multiply(np.pi,Mf)) # make this for all Freqs at once!
        logv = xp.log(v) #check if this should be log10 or ln
        v2 = xp.multiply(v,v)
        v3 = xp.multiply(v2,v)
        v4 = xp.multiply(v3,v)
        v5 = xp.multiply(v4,v)
        v6 = xp.multiply(v5,v)
        v7 = xp.multiply(v6,v)
        v8 = xp.multiply(v7,v)
        
        Dphasing = xp.multiply(2.0 * pn_coeff.v[7], v7)
        Dphasing = xp.add(Dphasing, xp.multiply(xp.add(pn_coeff.v[6], 
            xp.multiply(pn_coeff.vlogv[6], xp.add(1., logv))), v6))
        Dphasing = xp.add(Dphasing, xp.multiply(pn_coeff.vlogv[5], v5))
        Dphasing = xp.add(Dphasing, xp.multiply(-1. * pn_coeff.v[4], v4))
        Dphasing = xp.add(Dphasing, xp.multiply(-2. * pn_coeff.v[3], v3))
        Dphasing = xp.add(Dphasing, xp.multiply(-3. * pn_coeff.v[2], v2))
        Dphasing = xp.add(Dphasing, xp.multiply(-4. * pn_coeff.v[1], v))
        Dphasing = xp.add(Dphasing, -5. * pn_coeff.v[0])
        Dphasing = xp.divide(Dphasing, xp.multiply(3., v8))
        Dphasing = xp.multiply(Dphasing, np.pi)
                        
        return Dphasing

    def TF2_chirpTime(self, fseries_Hz, m1, m2, chi1z, chi2z, WFdict):
        
        pn = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, chi1z, chi2z, WFdict)
        
        Mtot = m1+m2
        
        Mtot_s = Mtot*lal.MTSUN_SI
        Mf = xp.multiply(Mtot_s,fseries_Hz) #[below_fISCO]
        #make this take fSeries as input
        
        #v_ISCO = xp.divide(1., xp.sqrt(6.))
        #f_ISCO = xp.divide(xp.power(v_ISCO, 3), xp.multiply(np.pi, Mtot_s))
        f_ISCO = xp.divide(self.fISCO_constant, Mtot_s)
        # store all but Mtot_s as a constant, 
        
        #print(Mf)
        
        dPhi = self.TF2_PhaseDerivative(Mf, pn)
        dPhi_ISCO = self.TF2_PhaseDerivative(Mtot_s*f_ISCO, pn)

        dPhi_diff = xp.subtract(dPhi_ISCO, dPhi)
        
        ChirpTimeSec = xp.divide(xp.multiply(dPhi_diff,Mtot_s),(2*np.pi))
        
        return ChirpTimeSec




















