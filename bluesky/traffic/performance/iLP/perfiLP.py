""" iLP aircraft performance calculations."""
import numpy as np
import bluesky as bs
from bluesky.traffic.performance.perfbase import PerfBase
from bluesky import settings
from bluesky.traffic.performance.bada import coeff_bada
from bluesky.tools.aero import kts, ft, g0, vtas2cas, vcas2tas
from bluesky.traffic.performance.iLP.performance import esf, phases, calclimits, PHASE
from bluesky.traffic.performance.iLP.iLP_ESTIMATOR import EEI

#Register settings default
settings.set_variable_defaults(perf_path_bada ='bluesky/resources/performance/BADA',
                               performance_dt = 1.0, verbose= False)\

settings.set_variable_defaults(performance_model='iLP', asas_dt=1.0)

if not coeff_bada.check():
    raise ImportError('íLP performance model: Error trying to find BADA files in ' + settings.perf_path_bada + '!')

class iLP(PerfBase):
    """
    iLabs Aircraft Performance (iLP) Model

    Methods:
        create(): initialize new aircraft with performance parameters
        update(): update performance parameters

    Created by  : wila
    Note: This class is based on
        BADA 3.10

    Edited by   : Albert
    Date: May-June 2025
    """
    def __init__(self):
        super().__init__()
        # Load coefficient files if we haven't yet
        coeff_bada.init(settings.perf_path_bada)

        self.warned = False                     # Flag: Did we warn for default perf parameters yet?
        self.warned2 = False                    # Flag: Use of piston engine aircraft?

        # Flight performance scheduling
        self.warned2 = False                    # Flag: Did we warn for default engine parameters yet?

        # Register the per-aircraft parameter arrays
        with self.settrafarrays():
            # engine
            self.jet        = np.array([])
            self.turbo      = np.array([])
            self.piston     = np.array([])

            # masses and dimensions
            self.mass       = np.array([])      # effective mass [kg]
            self.mref       = np.array([])      # ref. mass [kg]: 70% between min and max. mass
            self.mmin       = np.array([])      # OEW (or assumption) [kg]
            self.mmax       = np.array([])      # MTOW (or assumption) [kg]
            self.mpyld      = np.array([])      # MZFW-OEW (or assumption) [kg]
            self.gw         = np.array([])      # weight gradient on max. alt [m/kg]
            self.Sref       = np.array([])      # wing reference surface area [m^2]

            # flight enveloppe
            self.vmto       = np.array([])      # min TO spd [m/s]
            self.vmic       = np.array([])      # min climb spd [m/s]
            self.vmcr       = np.array([])      # min cruise spd [m/s]
            self.vmap       = np.array([])      # min approach spd [m/s]
            self.vmld       = np.array([])      # min landing spd [m/s]

            self.vmo        = np.array([])      # max operating speed [m/s]
            self.mmo        = np.array([])      # max operating mach number [-]
            self.hmax       = np.array([])      # max. alt above standard MSL (ISA) at MTOW [m]
            self.hmaxact    = np.array([])      # max. alt depending on temperature gradient [m]
            self.hmo        = np.array([])      # max. operating alt abov standard MSL [m]
            self.gt         = np.array([])      # temp. gradient on max. alt [ft/k]
            self.maxthr     = np.array([])      # maximum thrust [N]

            # Buffet Coefficients
            self.clbo       = np.array([])      # buffet onset lift coefficient [-]
            self.k          = np.array([])      # buffet coefficient [-]
            self.cm16       = np.array([])      # CM16

            # reference CAS speeds
            self.cascl      = np.array([])      # climb [m/s]
            self.cascr      = np.array([])      # cruise [m/s]
            self.casdes     = np.array([])      # descent [m/s]
            self.cascl2     = np.array([])      # climb 2 [m/s]
            self.cascr2     = np.array([])      # cruise 2 [m/s]
            self.casdes2    = np.array([])      # descent 2 [m/s]

            # reference mach numbers [-]
            self.macl       = np.array([])      # climb
            self.macr       = np.array([])      # cruise
            self.mades      = np.array([])      # descent

            # parasitic drag coefficients per phase [-]
            self.cd0to      = np.array([])      # phase takeoff
            self.cd0ic      = np.array([])      # phase initial climb
            self.cd0cr      = np.array([])      # phase cruise
            self.cd0ap      = np.array([])      # phase approach
            self.cd0ld      = np.array([])      # phase land
            self.gear       = np.array([])      # drag due to gear down

            # induced drag coefficients per phase [-]
            self.cd2to      = np.array([])      # phase takeoff
            self.cd2ic      = np.array([])      # phase initial climb
            self.cd2cr      = np.array([])      # phase cruise
            self.cd2ap      = np.array([])      # phase approach
            self.cd2ld      = np.array([])      # phase land

            # max climb thrust coefficients
            self.ctcth1     = np.array([])      # jet/piston [N], turboprop [ktN]
            self.ctcth2     = np.array([])      # [ft]
            self.ctcth3     = np.array([])      # jet [1/ft^2], turboprop [N], piston [ktN]

            # reduced climb power coefficient
            self.cred       = np.array([])      # [-]

            # 1st and 2nd thrust temp coefficient
            self.ctct1      = np.array([])      # [k]
            self.ctct2      = np.array([])      # [1/k]
            self.dtemp      = np.array([])      # [k]

            # Descent Fuel Flow Coefficients
            # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
            self.ctdesl     = np.array([])      # low alt descent thrust coefficient [-]
            self.ctdesh     = np.array([])      # high alt descent thrust coefficient [-]
            self.ctdesa     = np.array([])      # approach thrust coefficient [-]
            self.ctdesld    = np.array([])      # landing thrust coefficient [-]

            # transition altitude for calculation of descent thrust
            self.hpdes      = np.array([])      # [m]

            # Energy Share Factor
            self.ESF        = np.array([])      # [-]

            # reference speed during descent
            self.vdes       = np.array([])      # [m/s]
            self.mdes       = np.array([])      # [-]

            # flight phase
            self.phase      = np.array([])
            self.post_flight = np.array([])     # taxi prior of post flight?
            self.pf_flag    = np.array([])

            # Thrust specific fuel consumption coefficients
            self.cf1        = np.array([])      # jet [kg/(min*kN)], turboprop [kg/(min*kN*knot)], piston [kg/min]
            self.cf2        = np.array([])      # [knots]
            self.cf3        = np.array([])      # [kg/min]
            self.cf4        = np.array([])      # [ft]
            self.cf_cruise  = np.array([])      # [-]

            # performance
            self.T          = np.array([])      # thrust
            self.D          = np.array([])      # drag
            self.fuelflow   = np.array([])      # fuel flow

            # ground
            self.tol        = np.array([])      # take-off length[m]
            self.ldl        = np.array([])      # landing length[m]
            self.ws         = np.array([])      # wingspan [m]
            self.len        = np.array([])      # aircraft length[m]
            self.gr_acc     = np.array([])      # ground acceleration [m/s^2]
            self.gr_dec     = np.array([])      # ground deceleration [m/s^2]

            # # limit settings
            self.limspd     = np.array([])      # limit speed
            self.limspd_flag = np.array([], dtype=bool)  # flag for limit spd - we have to test for max and min
            self.limalt     = np.array([])      # limit altitude
            self.limalt_flag = np.array([])     # A need to limit altitude has been detected
            self.limvs      = np.array([])      # limit vertical speed due to thrust limitation
            self.limvs_flag = np.array([])      # A need to limit V/S detected

        # Default bank angles per flight phase
        self.bphase = np.deg2rad(np.array([15, 35, 35, 35, 35, 35, 15, 45]))


    def engchange(self, acid, engine_id = None):
        return False, "BADA performance model doesn't allow changing engine type"

    def create(self, n=1):
        super().create(n)
        """
                Create new aircraft
                note: coefficients are initialized in SI units
                """

        actypes = bs.traf.type[-n:]

        # general
        # descignate aircraft to its aircraft type
        for i, actype in enumerate(actypes):
            syn, coeff = coeff_bada.getCoefficients(actype)
            if syn:
                continue
            syn, coeff = coeff_bada.getCoefficients('B744')
            bs.traf.type[-n + i] = syn.accode

            if not settings.verbose:
                if not self.warned:
                    print("Aircraft is using default B747-400 performance.")
            else:
                print("Flight " + bs.traf.id[-n:][-1] + " has an unknown aircraft type, " + actype +
                      ", BlueSky then uses default B747-400 performance.")

        # designate aircraft to its aircraft type
        self.jet[-n:] = 1 if coeff.engtype == 'Jet' else 0
        self.turbo[-n:] = 1 if coeff.engtype == 'Turboprop' else 0
        self.piston[-n:] = 1 if coeff.engtype == 'Pisto' else 0

        # Initial aircraft mass is currently reference mass.
        # BADA 3.12 also supports masses between 1.2*mmin and mmax
        self.mass[-n:] = coeff.m_ref * 1000.0
        self.mref[-n:] = coeff.m_ref * 1000.0
        self.mmin[-n:] = coeff.m_min * 1000.0
        self.mmax[-n:] = coeff.m_max * 1000.0
        self.gw[-n:] = coeff.mass_grad * ft

        # Surface Area [m^2]
        self.Sref[-n:] = coeff.S

        # flight envelope
        # minimum speed per phase
        self.vmto[-n:] = coeff.Vstall_to * coeff.CVmin_to * kts
        self.vmcr[-n:] = coeff.Vstall_cr * coeff.CVmin * kts
        self.vmap[-n:] = coeff.Vstall_ap * coeff.CVmin * kts
        self.vmld[-n:] = coeff.Vstall_ld * coeff.CVmin * kts
        self.vmin[-n:] = 0.0
        self.vmo[-n:] = coeff.VMO * kts
        self.mmo[-n:] = coeff.MMO
        self.vmax[-n:] = self.vmo[-n:]

        # max. altitude parameters
        self.hmo[-n:] = coeff.h_MO * ft
        self.hmax[-n:] = coeff.h_max * ft
        self.hmaxact[-n:] = coeff.h_max * ft  # initialize with hmax
        self.gt[-n:] = coeff.temp_grad * ft

        # max thrust setting
        self.maxthr[-n:] = 1e6  # initialize with excessive setting to avoid unrealistic limit setting

        # Buffet Coefficients
        self.clbo[-n:] = coeff.Clbo
        self.k[-n:] = coeff.k
        self.cm16[-n:] = coeff.CM16

        # reference speeds
        # reference CAS speeds
        self.cascl[-n:] = coeff.CAScl1[0] * kts
        self.cascr[-n:] = coeff.CAScr1[0] * kts
        self.casdes[-n:] = coeff.CASdes1[0] * kts

        self.cascl2[-n:] = coeff.CAScl2[0] * kts
        self.cascr2[-n:] = coeff.CAScr2[0] * kts
        self.casdes2[-n:] = coeff.CASdes2[0] * kts

        # reference mach numbers
        self.macl[-n:] = coeff.Mcl[0]
        self.macr[-n:] = coeff.Mcr[0]
        self.mades[-n:] = coeff.Mdes[0]

        # reference speed during descent
        self.vdes[-n:] = coeff.Vdes_ref * kts
        self.mdes[-n:] = coeff.Mdes_ref

        # aerodynamics
        # parasitic drag coefficients per phase
        self.cd0to[-n:] = coeff.CD0_to
        self.cd0ic[-n:] = coeff.CD0_ic
        self.cd0cr[-n:] = coeff.CD0_cr
        self.cd0ap[-n:] = coeff.CD0_ap
        self.cd0ld[-n:] = coeff.CD0_ld
        self.gear[-n:] = coeff.CD0_gear

        # induced drag coefficients per phase
        self.cd2to[-n:] = coeff.CD2_to
        self.cd2ic[-n:] = coeff.CD2_ic
        self.cd2cr[-n:] = coeff.CD2_cr
        self.cd2ap[-n:] = coeff.CD2_ap
        self.cd2ld[-n:] = coeff.CD2_ld

        # reduced climb coefficient
        self.cred[-n:] = np.where(
            self.jet[-n:], coeff.Cred_jet,
            np.where(self.turbo[-n:], coeff.Cred_turboprop, coeff.Cred_piston)
        )

        # performance

        # max climb thrust coefficients
        self.ctcth1[-n:] = coeff.CTC[0]  # jet/piston [N], turboprop [ktN]
        self.ctcth2[-n:] = coeff.CTC[1]  # [ft]
        self.ctcth3[-n:] = coeff.CTC[2]  # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient
        self.ctct1[-n:] = coeff.CTC[3]  # [k]
        self.ctct2[-n:] = coeff.CTC[4]  # [1/k]
        self.dtemp[-n:] = 0.0  # [k], difference from current to ISA temperature. At the moment: 0, as ISA environment

        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl[-n:] = coeff.CTdes_low
        self.ctdesh[-n:] = coeff.CTdes_high
        self.ctdesa[-n:] = coeff.CTdes_app
        self.ctdesld[-n:] = coeff.CTdes_land

        # transition altitude for calculation of descent thrust
        self.hpdes[-n:] = coeff.Hp_des * ft
        self.ESF[-n:] = 1.0  # neutral initialisation

        # flight phase
        self.phase[-n:] = PHASE["None"]
        self.post_flight[-n:] = False  # we assume prior
        self.pf_flag[-n:] = True

        # Thrust specific fuel consumption coefficients
        # prevent from division per zero in fuelflow calculation
        self.cf1[-n:] = coeff.Cf1
        self.cf2[-n:] = 1.0 if abs(coeff.Cf2) < 1e-9 else coeff.Cf2
        self.cf3[-n:] = coeff.Cf3
        self.cf4[-n:] = 1.0 if abs(coeff.Cf4) < 1e-9 else coeff.Cf4
        self.cf_cruise[-n:] = coeff.Cf_cruise

        self.thrust[-n:] = 0.0
        self.D[-n:] = 0.0
        self.fuelflow[-n:] = 0.0

        # ground
        self.tol[-n:] = coeff.TOL
        self.ldl[-n:] = coeff.LDL
        self.ws[-n:] = coeff.wingspan
        self.len[-n:] = coeff.length

        # for now, BADA aircraft have the same acceleration as deceleration
        self.gr_acc[-n:] = coeff.gr_acc
        self.gr_dec[-n:] = -coeff.gr_acc

    def update(self, dt):
        ''' Periodic update function for performance calculations. '''
        # BADA version
        swbada = True
        delalt = bs.traf.selalt - bs.traf.alt

        # update phase
        self.phase, self.bank = phases(bs.traf.alt, bs.traf.vs, bs.traf.cas, bs.traf.ap.bankdef, self.bphase, bs.traf.swhdgsel)

        # AERODYNAMICS
        # Lift
        qS = 0.5 * bs.traf.rho * np.maximum(1., bs.traf.tas) * np.maximum(1., bs.traf.tas) * self.Sref
        cl = (self.mass * g0/ (qS * np.cos(self.bank)) * (self.phase != PHASE["GD"]) +
              0.* (self.phase == PHASE['GD']))

        # Drag Coefficients

        # phases CL, CR, SC, SD, DE
        cdph = self.cd0cr + self.cd2cr * (cl * cl)

        # phases TO
        # in case take-off coefficient in OPF-Files are set to zero:
        # Use cruise value instead
        cdto = np.where(self.cd0to != 0, self.cd0to + self.gear + self.cd2to * (cl * cl) , cdph + self.gear)  #TODO: Add to TU Delft BADA version

        # phases IC
        # in case initial climb coefficient in OPF-Files are set to zero:
        # Use cruise value instead
        cdic = np.where(self.cd0ic != 0, self.cd0ic + self.cd2ic * (cl * cl), cdph) #TODO: Add to TU Delft BADA version

        # phases AP
        # in case approach coefficient in OPF-Files are set to zero:
        # Use cruise value instead
        cdap = np.where(self.cd0ap != 0, self.cd0ap + self.cd2ap * (cl * cl), cdph)

        # phases LD
        # in case landing coefficient in OPF-Files are set to zero:
        # Use cruise value instead
        cdld = np.where(self.cd0ld != 0, self.cd0ld + self.gear + self.cd2ld * (cl * cl), cdph + self.gear)

        # now combine phases
        cd = ((self.phase == PHASE['TO']) * cdto + (self.phase == PHASE['IC']) * cdic +
              (self.phase == PHASE['CL']) * cdph + (self.phase == PHASE['CR']) * cdph +
              (self.phase == PHASE['SC']) * cdph + (self.phase == PHASE['SD']) * cdph +
              (self.phase == PHASE['DE']) * cdph + (self.phase == PHASE['AP']) * cdap +
              (self.phase == PHASE['LD']) * cdld )

        # Drag
        self.D = cd * qS

        # Energy Share Factor and Crossover Altitude

        # Energy Share Factor conditions
        epsalt = np.array([0.001] * bs.traf.ntraf)
        climb = np.array(delalt > epsalt)
        descent = np.array(delalt < -epsalt)
        level = np.array(np.abs(delalt) < 0.0001) * 1

        # energy share factor
        delspd = bs.traf.aporasas.tas - bs.traf.tas
        selmach = bs.traf.selspd < 2.0

        self.ESF = esf(bs.traf.alt, bs.traf.M, climb, descent, delspd, selmach)

        # THRUST
        # 1. climb: max.climb thrust in ISA conditions (p. 32, BADA User Manual 3.12)
        # condition: delta altitude positive
        # jet
        # condition
        cljet = np.logical_and.reduce([climb, self.jet]) * 1

        # thrust
        Tj = self.ctcth1 * (
                    1 - (bs.traf.alt / ft) / self.ctcth2 + self.ctcth3 * (bs.traf.alt / ft) * (bs.traf.alt / ft))

        # combine jet and default aircraft
        Tjc = cljet * Tj
        # turboprop
        # condition
        clturbo = np.logical_and.reduce([climb, self.turbo]) * 1

        # thrust
        Tt = self.ctcth1 / np.maximum(1., bs.traf.tas / kts) * (1 - (bs.traf.alt / ft) / self.ctcth2) + self.ctcth3

        # merge
        Ttc = clturbo * Tt

        # piston
        clpiston = np.logical_and.reduce([climb, self.piston]) * 1
        Tp = self.ctcth1 * (1 - (bs.traf.alt / ft) / self.ctcth2) + self.ctcth3 / np.maximum(1., bs.traf.tas / kts)
        Tpc = clpiston * Tp

        # max climb thrust for futher calculations (equals maximum avaliable thrust)
        maxthr = Tj * self.jet + Tt * self.turbo + Tp * self.piston

        # ADDED; update max thrust
        self.maxthr = maxthr

        # 2. level flight: Thr = D.
        Tlvl = level * self.D

        # 3. Descent: condition: vs negative/ H>hdes: fixed formula. H<hdes: phase de, ap, ld

        # above or below Hpdes? Careful! If non-ISA: ALT must be replaced by Hp!
        delh = (bs.traf.alt - self.hpdes)

        # above Hpdes:
        high = np.array(delh > 0)
        Tdesh = maxthr * self.ctdesh * np.logical_and.reduce([descent, high])

        # below Hpdes
        low = np.array(delh < 0)

        # phase cruise
        TdeslCR = maxthr * self.ctdesl * np.logical_and.reduce([descent, low, (self.phase == PHASE['CR'])])
        # phase descent
        TdeslDE = maxthr * self.ctdesl * np.logical_and.reduce([descent, low, (self.phase == PHASE['DE'])])
        # phase approach
        TdeslAP = maxthr * self.ctdesa * np.logical_and.reduce([descent, low, (self.phase == PHASE['AP'])])
        # phase landing
        TdeslLD = maxthr * self.ctdesld * np.logical_and.reduce([descent, low, (self.phase == PHASE['LD'])])
        # phase ground: minimum descent thrust as a first approach
        Tgd = np.minimum.reduce([Tdesh, TdeslLD]) * (self.phase == PHASE['GD'])

        # merge all thrust conditions
        T = np.maximum.reduce([Tjc, Ttc, Tpc, Tlvl, Tdesh, TdeslCR, TdeslDE, TdeslAP, TdeslLD, Tgd])

        # vertical speed. Note: ISA only ( tISA = 1 )
        # for climbs: reducing factor (reduced climb power) is multiplie
        # cred applies below 0.8*hmax and for climbing aircraft only
        hcred = np.array(bs.traf.alt < (self.hmaxact*0.8))
        clh = np.logical_and.reduce([hcred, climb])
        cred = self.cred*clh
        cpred = 1-cred*((self.mmax-self.mass)/(self.mmax-self.mmin))

        # switch for given vertical speed selvs
        if (bs.traf.selvs.any() > 0.) or (bs.traf.selvs.any() < 0.):
            # thrust = f(selvs)
            T = ((np.sign(bs.traf.az) * bs.traf.aporasas.vs * self.mass * g0) /
                 (self.ESF * np.maximum(bs.traf.eps, bs.traf.tas) * cpred)) + self.D

        self.thrust = np.where(T > maxthr - 1, maxthr - 1, T)

        # Fuel consumption
        # thrust specific fuel consumption - jet
        # thrust
        etaj = self.cf1 * (1.0 + (bs.traf.tas / kts) / self.cf2)
        # merge
        ej = etaj * self.jet

        # thrust specific fuel consumption - turboprop

        # thrust
        etat = self.cf1 * (1. - (bs.traf.tas / kts) / self.cf2) * ((bs.traf.tas / kts) / 1000.)
        # merge
        et = etat * self.turbo

        # thrust specific fuel consumption for all aircraft
        # eta is given in [kg/(min*kN)] - convert to [kg/(min*N)]
        eta = np.maximum.reduce([ej, et]) / 1000.

        # nominal fuel flow - (jet & turbo) and piston
        # condition jet,turbo:
        jt = np.maximum.reduce([self.jet, self.turbo])
        pdf = np.maximum.reduce([self.piston])

        fnomjt = eta * self.thrust * jt
        fnomp = self.cf1 * pdf
        # merge
        fnom = fnomjt + fnomp

        # minimal fuel flow jet, turbo and piston
        fminjt = self.cf3 * (1 - (bs.traf.alt / ft) / self.cf4) * jt
        fminp = self.cf3 * pdf
        # merge
        fmin = fminjt + fminp

        # cruise fuel flow jet, turbo and piston
        fcrjt = eta * self.thrust * self.cf_cruise * jt
        fcrp = self.cf1 * self.cf_cruise * pdf
        # merge
        fcr = fcrjt + fcrp

        # approach/landing fuel flow
        fal = np.maximum(fnom, fmin)

        # designate each aircraft to its fuelflow
        # takeoff
        ffto = fnom * (self.phase == PHASE['TO'])

        # initial climb
        ffic = fnom * (self.phase == PHASE['IC']) / 2

        # phase cruise and climb
        cc = np.logical_and.reduce([climb, (self.phase == PHASE['CR'])]) * 1
        ffcc = fnom * cc

        # cruise and level
        ffcrl = fcr * level

        # descent cruise configuration
        cd2 = np.logical_and.reduce([descent, (self.phase == PHASE['CR'])]) * 1
        ffcd = cd2 * fmin

        # approach
        ffap = fal * (self.phase == PHASE['AP'])

        # landing
        ffld = fal * (self.phase == PHASE['LD'])

        # ground
        ffgd = fmin * (self.phase == PHASE['GD'])

        # fuel flow for each condition
        self.fuelflow = np.maximum.reduce(
            [ffto, ffic, ffcc, ffcrl, ffcd, ffap, ffld, ffgd]) / 60.  # convert from kg/min to kg/sec

        # update mass
        self.mass -= self.fuelflow * dt  # Use fuelflow in kg/min

        # for aircraft on the runway and taxiways we need to know, whether they
        # are prior or after their flight
        self.post_flight = np.where(descent, True, self.post_flight)

        # when landing, we would like to stop the aircraft.
        bs.traf.aporasas.tas = np.where((bs.traf.alt < 0.5) * (self.post_flight) * self.pf_flag, 0.0,
                                        bs.traf.aporasas.tas)

        # otherwise taxiing will be impossible afterwards
        # pf_flag is released so post_flight flag is only triggered once

        self.pf_flag = np.where((bs.traf.alt < 0.5) * (self.post_flight), False, self.pf_flag)

        # define acceleration: aircraft taxiing and taking off use ground acceleration,
        # landing aircraft use ground deceleration, others use standard acceleration
        # --> BADA uses the same value for ground acceleration as for deceleration
        self.axmax = ((self.phase == PHASE['IC']) + (self.phase == PHASE['CL']) + (self.phase == PHASE['CR']) +
                      (self.phase == PHASE['DE']) + (self.phase == PHASE['AP']) + (self.phase == PHASE['LD'])) * 0.5 \
                     + ((self.phase == PHASE['TO']) + (self.phase == PHASE['GD']) * (1-self.post_flight)) * self.gr_acc\
                     + (self.phase == PHASE['GD']) * self.post_flight * self.gr_acc

        # self.axmax = np.where(bs.traf.ap.geodescent, 0.2, self.axmax)  # 0.2 kts decel during geo descent

    def limits(self, intent_v, intent_vs, intent_h, ax):
        """FLIGHT ENVELPOE"""
        # summarize minimum speeds - ac in ground mode might be pushing back
        self.vmin = (self.phase == PHASE['TO']) * self.vmto + (self.phase == PHASE['IC']) * self.vmic + \
                    (self.phase == PHASE['CL']) * self.vmcr + (self.phase == PHASE['CR']) * self.vmcr + \
                    (self.phase == PHASE['DE']) * self.vmcr * (self.phase == PHASE['AP']) * self.vmap + \
                    (self.phase == PHASE['LD']) * self.vmld + (self.phase == PHASE['GD']) * -10.

        # maximum altitude: hmax/act = MIN[hmo, hmax+gt*(dtemp-ctc1)+gw*(mmax-mact)]
        #                   or hmo if hmx ==0 ()
        # at the moment just ISA atmosphere, dtemp  = 0
        c1 = self.dtemp - self.ctct1

        # if c1<0: c1 = 0
        # values above 0 remain, values below are replaced through 0
        c1m = np.array(c1 < 0) * 0.00000000000001
        c1def = np.maximum(c1, c1m)

        self.hact = self.hmax + self.gt * c1def + self.gw * (self.mmax - self.mass)
        # if hmax in OPF File ==0: hmaxact = hmo, else minimum(hmo, hmact)
        self.hmaxact = (self.hmax == 0) * self.hmo + (self.hmax != 0) * np.minimum(self.hmo, self.hact)

        # forwarding to tools
        self.limspd, self.limspd_flag, self.limalt, \
            self.limalt_flag, self.limvs, self.limvs_flag = calclimits(
            vtas2cas(intent_v, bs.traf.alt), bs.traf.gs,
            self.vmto, self.vmin, self.vmo, self.mmo,
            bs.traf.M, bs.traf.alt, self.hmaxact,
            intent_h, intent_vs, self.maxthr,
            self.thrust, self.D, bs.traf.tas,
            self.mass, self.ESF, self.phase)

        # Update desired sates with values within the flight envelope
        # When CAS is limited, it needs to be converted to TAS as only this TAS is used later on!
        allowed_tas = np.where(self.limspd_flag, vcas2tas(
            self.limspd, bs.traf.alt), intent_v)

        # Autopilot selected altitude [m]
        allowed_alt = np.where(self.limalt_flag, self.limalt, intent_h)

        # Autopilot selected vertical speed (V/S)
        allowed_vs = np.where(self.limvs_flag, self.limvs, intent_vs)

        return allowed_tas, allowed_vs, allowed_alt

    def show_performance(self, acid):
        # PERF acid command
        bs.scr.echo("Flight phase: %s" % self.phase[acid])
        bs.scr.echo("Thrust: %d kN" % (self.thrust[acid] / 1000))
        bs.scr.echo("Drag: %d kN" % (self.D[acid] / 1000))
        bs.scr.echo("Fuel flow: %.2f kg/s" % self.fuelflow[acid])
        bs.scr.echo(
            "Speed envelope: Min: %d MMO: %d kts M %.2f" % (int(self.vmin[acid] / kts), int(self.vmo[acid] / kts),
                                                            self.mmo[acid]))
        bs.scr.echo("Ceiling: %d ft" % (int(self.hmax[acid] / ft)))
        return True