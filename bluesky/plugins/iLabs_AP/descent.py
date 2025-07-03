"""
Calculate the attributes needed to determine the descent path of an aircraft, such as ROD and TAS for 100 ft segments

Created by: Winand Mathoera
Date: 31/12/2022
Adapted by: Teun Vleming
Date: 02/2024
Adapted by: Albert Chou
Date: 07/2025
"""

import numpy as np
import bluesky as bs
from bluesky.tools.aero import ft, kts, gamma, gamma1, gamma2, R, beta, nm, fpm, vcasormach2tas, vcas2tas, vtas2cas, \
    cas2tas, vcas2mach, g0, vmach2cas, density, vatmos, vcasormach
from bluesky.traffic.performance.iLP.perfiLP import esf
from bluesky.traffic.performance.iLP.performance import PHASE

class Descent:
    """
    Definition: find speed, VS, gamma for a certain altitude range in 100ft intervals. Used to find descent distance.
    Methods:
        __init__(): produce all needed attributes from other functions

    """
    def __init__(self, idx, start_alt, end_alt, concas = -1, seg_height = 100):
        """
        Function: Call the different functions in the class to calculate the descent path attributes
        Simplifications:
            Mass is kept constant for a calculation
            Altitude intervals of 100 ft are taken

        Args:
            idx:        a/c index
            start_alt:  top of descent [m]
            end_alt:    bottom of descent [m]
            concas:     constant calibrated airspeed if given, else use -1 to dismiss
        """

        # array of altitude segments
        self.segments = np.arange(end_alt, start_alt, seg_height * ft)

        # determine speed from speedschedule in each segment, reject if constant CAS is given
        self.speed = np.array([self.descent_speedschedule(idx, i) for i in self.segments])

        if concas > 0:
            self.speed = np.full(np.shape(self.segments), concas)

        # Convert descent speed schedule to TAS
        self.tasspeeds = vcasormach2tas(self.speed, self.segments)

        # Determine Phase for each segment
        self.phase = np.array([self.phases(i, end_alt, bs.traf.ap.calt[idx]) for i in self.segments])

        # Determine the Energy Share Factor (ESF) in each segment
        self.esft = np.array([self.esf(i, self.speed[index])
                              for index, i in enumerate(self.segments)])

        # Determine Thrust and Drag in each segment and separate into two arrays
        td = np.array([self.TandD(idx, i, self.tasspeeds[index], self.phase[index]) for
                       index, i in enumerate(self.segments)], dtype = object)
        self.thrust, self.drag, data = zip(*td)

        # Determine rate of descent (RoD) using total energy equation
        self.rod = ((np.array(self.thrust) - np.array(self.drag)) * self.tasspeeds) / (bs.traf.perf.mass[idx] * g0) * self.esft

        # Determine descent angle
        self.gamma = np.arcsin(self.rod / self.tasspeeds)

        # determine the speeds below and above the deceleration altitudes based on speed schedule
        self.decel_alts = np.array([999 * ft, 1499 * ft, 1999 * ft, 2999 * ft, 5999 * ft, 9999* ft, 26000 * ft])
        self.vsegm = np.array([self.descent_speedschedule(idx, i) for i in self.decel_alts] +
                              [self.descent_speedschedule(idx, 26000 * ft * 1.1)])

        # Determine at which altitude a decceleration segment is needed
        self.decel_altspd = []
        # format : (alt, new speed, old speed)
        for index, i in enumerate(self.decel_alts):
            # skip higher altitudes
            if i > start_alt or i < end_alt:
                continue
            self.decel_altspd.append((i, self.vsegm[index], self.vsegm[index + 1]))


    def descent_speedschedule(self, idx,alt):
        # TODO: Make this compatible with OpenAP
        '''
        Function: determine CAS based on speed schedule from the BADA manual
        Args:
            idx:    a/c id
            alt:    a/c altitude [m]
        '''

        alt = alt / ft

        # Correction for minimum landing speed
        corr = np.sqrt(bs.traf.perf.mass[idx] / bs.traf.perf.mref[idx])

        # Calculate the speeds for each altitude segment based on either
        # the minimum landing speed or the standard descent speed
        l1 = corr * bs.traf.perf.vmld[idx]  / kts + 5
        l2 = corr * bs.traf.perf.vmld[idx]  / kts + 10
        l3 = corr * bs.traf.perf.vmld[idx]  / kts + 20
        l4 = corr * bs.traf.perf.vmld[idx]  / kts + 50
        l5 = np.where(bs.traf.perf.casdes[idx]  / kts <= 220, bs.traf.perf.casdes[idx]  / kts, 220)
        l6 = np.where(bs.traf.perf.casdes2[idx]  / kts <= 250, bs.traf.perf.casdes2[idx]  / kts, 250)
        l7 = 280
        l8 = np.where(bs.traf.perf.mmax[idx]  > 7000, 0.78, 0.82)

        # Piston aircraft speeds
        # TODO: Check the Vdes_i value and casdes values if they are in kts or m/s
        l9 = corr * bs.traf.perf.vmld[idx] / kts + 0
        l10 = corr * bs.traf.perf.vmld[idx]  / kts + 0
        l11 = corr * bs.traf.perf.vmld[idx]  / kts + 0
        l12 = bs.traf.perf.casdes[idx]
        l13 = bs.traf.perf.casdes2[idx]
        l14 = bs.traf.perf.mades[idx]

        # Find the segment of Turboprop and Turbofan aircraft and obtain correct speed for that segment
        spds = np.where(alt <= 999, 1, 0) * l1 * kts + \
               np.logical_and(alt > 999, alt <= 1499) * l2 * kts + \
               np.logical_and(alt > 1499, alt <= 1999) * l3 * kts + \
               np.logical_and(alt > 1999, alt <= 2999) * l4 * kts + \
               np.logical_and(alt > 2999, alt <= 5999) * l5 * kts + \
               np.logical_and(alt > 5999, alt <= 9999) * l6 * kts + \
               np.logical_and(alt > 9999, alt <= 26000) * l7 * kts + \
               np.where(alt > 26000, 1, 0) * l8

        # Segments for piston aircraft
        spds_p = np.where(alt <= 499, 1, 0) * l9 * kts + \
                 np.logical_and(alt > 499, alt <= 999) * l10 * kts + \
                 np.logical_and(alt > 999, alt <= 1499) * l11 * kts + \
                 np.logical_and(alt > 1499, alt <= 9999) * l12 * kts + \
                 np.logical_and(alt > 9999, alt <= bs.traf.perf.hpdes[idx] / 0.3048) * l13 * kts + \
                 np.where(alt > bs.traf.perf.hpdes[idx] / 0.3048, 1, 0) * l14

        spd = np.where(bs.traf.perf.piston[idx] == 1, spds_p, spds)

        return spd

    def phases(self, alt, end_alt, calt):
        '''
                Function: determine phase based on method taken from perfwlb.py, for single a/c
                Args:
                    alt:        a/c altitude
                    end_alt:    a/c end altitude
                    calt: defined cruise altitude
                '''
        # Determine difference in altitude
        del_alt = alt - end_alt

        # Descent logic will only be switched on when it reaches Cruise Phase

        # phase CR[4]: alt >= FL100,, delalt <= 0
        CR_alt = np.array(alt > (100. * 100 * ft))
        CR_calt = np.array(alt >= calt)

        cr = CR_alt * CR_calt * PHASE['CR']

        # phase DE[5]: alt >= FL100, delalt > 0
        DE_alt = np.array(alt > (100. * 100 * ft))
        DE_calt = np.array(alt < calt)
        DE_delalt = np.array(del_alt > 0)


        de = DE_alt * DE_calt * DE_delalt * PHASE['DE']

        # phase AP[6]: 2000 <= alt <= 10000, delalt > 0
        AP_alt = np.array((alt >= (2000. * ft)) & (alt <= (10000. * ft)))
        AP_delalt = np.array(del_alt > 0)

        ap = AP_alt * AP_delalt * PHASE['AP']

        # phase LD[7]: alt <=2000, del_alt > 0
        LD_alt = np.array(alt <= (2000. * ft))
        LD_delalt = np.array(del_alt > 0)

        ld = LD_alt * LD_delalt * PHASE['LD']

        # phase GD[8]: alt <= 1,
        GD_alt = np.array(alt <= (1. * ft))

        gd = GD_alt * PHASE['GD']
        return np.maximum.reduce([cr, de, ap, ld, gd])

    def esf(self, alt, spd):
        '''
        Function: determine energy share factor based on method taken from iLP performance.py, for single a/c
        Args:
        alt:    a/c altitude
        spd:    a/c CAS or Mach
        '''

        if spd < 3: M = spd
        else: M = vcas2mach(spd, alt)

        selmach = spd < 2.0

        abtp  = alt > 11000.0
        beltp = np.logical_not(abtp)
        selcas = np.logical_not(selmach)

        if selmach and abtp:
            return 1

        if selmach and beltp:
            return 1.0 / (1.0 + ((gamma * R * beta) / (2.0 * g0)) * M**2)

        if selcas and beltp:
            return 1.0 / (1.0 + (((gamma * R * beta) / (2.0 * g0)) * (M**2)) +
                          ((1.0 + gamma1 * (M**2))**(-1.0 / (gamma - 1.0))) *
                          (((1.0 + gamma1 * (M**2))**gamma2) - 1))

        if selcas and abtp:
            return 1.0 / (1.0 + ((1.0 + gamma1 * (M**2))**(-1.0 / (gamma - 1.0))) *
                          (((1.0 + gamma1 * (M**2))**gamma2) - 1.0))

    def TandD(self, idx, alt, tas, phase):
        """
        Function: determine thrust and drag based on method taken from periLP.py, for single a/c
        Args:
            idx:    a/c id
            alt:    a/c altitude
            tas:    a/c true airspeed
            phase:  a/c phase

        Returns:
            thrust [N]
            drag [N]
            data
        """
        mass = bs.traf.perf.mass[idx]

        p, rho, temp = vatmos(alt)

        qS = 0.5 * rho * np.maximum(1., tas) * np.maximum(1., tas) * bs.traf.perf.Sref[idx]
        cl = mass * g0 / qS

        # Transition altitude for calculation for descent thrust
        delh = alt - bs.traf.perf.hpdes[idx]

        # Determine max. climb thrust for different aircraft types
        # Jet
        Tj = (bs.traf.perf.ctcth1 *
              (1 - (alt / ft) / bs.traf.perf.ctcth2 + bs.traf.perf.ctcth3 * (alt / ft) * (alt / ft)))

        # Turboprop
        Tt = (bs.traf.perf.ctcth1[idx] / np.maximum(1., tas / kts) * (1 - (alt / ft) / bs.traf.perf.ctcth2[idx]) +
              bs.traf.perf.ctcth3[idx])

        # Piston
        Tp = (bs.traf.perf.ctcth1[idx] * (1 - (alt / ft) / bs.traf.perf.ctcth2[idx]) +
              bs.traf.perf.ctcth3[idx] / np.maximum(1., tas / kts))

        max_thrust = Tj * bs.traf.perf.jet[idx] + Tt * bs.traf.perf.turbo[idx] + Tp * bs.traf.perf.piston[idx]

        # drag coefficients for phases CR[4] and DE[5] phase are the same
        cd = bs.traf.perf.cd0cr[idx] + bs.traf.perf.cd2cr * (cl * cl)
        if delh > 0:    T = max_thrust * bs.traf.perf.ctdesh[idx]
        else:           T = max_thrust * bs.traf.perf.ctdesl[idx]

        # compute drag coefficient and thrust for phase AP[6]
        if phase == PHASE['AP']:
            cd = np.where(bs.traf.perf.cd0ap != 0, bs.traf.perf.cd0ap[idx] + bs.traf.perf.cd2ap[idx] * (cl * cl), cd)
            if delh <= 0:   T = max_thrust * bs.traf.perf.ctdesa[idx]

        # compute drag coefficient and thrust for phase LD[7]
        if phase == PHASE['LD']:
            cd = np.where(bs.traf.perf.cd0ld != 0, bs.traf.perf.cd0ld[idx] + bs.traf.perf.gear[idx] +
                          bs.traf.perf.cd2ld[idx] * (cl * cl), cd + bs.traf.perf.gear[idx])
            if delh <= 0:   T = max_thrust * bs.traf.perf.ctdesld[idx]

        D = cd * qS

        return T, D, [cd, rho, bs.traf.perf.cd0cr[idx], bs.traf.perf.cd2cr[idx], cl, qS]