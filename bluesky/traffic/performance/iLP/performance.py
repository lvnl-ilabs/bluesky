import numpy as np
from bluesky.tools.aero import ft, gamma, gamma1, gamma2, R, beta, g0, vmach2cas

NA = 0      # Unknown phase
TO = 1      # Take-off
IC = 2      # Initial climb
CL = 3      # Climb
CR = 4      # Cruise
SC = 41     # Step Climb (Cruise)
SD = 42     # Step Descent (Cruise)
DE = 5      # Descent
AP = 6      # Approach
LD = 7      # Landing
GD = 8      # Ground

PHASE = {"None":0,
         "TO"  :1,      # Take-off
         "IC"  :2,      # Initial climb
         "CL"  :3,      # Climb
         "CR"  :4,      # Cruise
         "SC"  :41,     # Step Climb (Cruise)
         "SD"  :42,     # Step Descent (Cruise)
         "DE"  :5,      # Descent
         "AP"  :6,      # Approach
         "LD"  :7,      # Landing
         "GD"  :8,      # Ground
         "to"  :1,
         "ic"  :2,
         "cl"  :3,
         "cr"  :4,
         "sc"  :41,
         "sd"  :42,
         "de"  :5,      # and lower case to be sure
         "ap"  :6,
         "ld"  :7,
         "gd"  :8,
        }

fpm = ft / 60.
#-------------------------------------------------------------------------------------------------------
# FLIGHT PHASES
# Adaption based on the combination of BADA 3.12 User Manual, chapter 3.5, p. 19 and OpenAP
#-------------------------------------------------------------------------------------------------------

def phases(alt, roc, spd, bank, bphase, swhdgsel):
    # flight phases: TO (1), IC (2), CL (3), CR(4), DE(5), AP(6), LD(7), GD(8)
    # -------------------------------------------------
    # phase TO[1]: alt <= 75, roc >= 0
    TO_alt = np.array(alt <= (75. * ft))
    TO_roc = np.array(roc >= 0.)

    to = TO_alt * TO_roc * TO

    #-------------------------------------------------
    # phase IC[2]: 75 <= alt <= 2000, roc >= 0
    IC_alt = np.array((alt > (75. * ft)) & (alt <= (2000. * ft)))
    IC_roc = np.array(roc >= (0 * fpm))

    ic = IC_alt * IC_roc * IC

    # -------------------------------------------------
    # phase CL[3]: alt >= 2000, roc >= 0
    CL_alt = np.array(alt > (2000. * ft))
    CL_roc = np.array(roc > (0 * fpm))

    cl = CL_alt * CL_roc * CL

    # -------------------------------------------------
    # phase CR[4]: alt >= FL100, -150 <= roc <= 150
    CR_alt = np.array(alt > (100. * 100 * ft))
    CR_roc = np.array((roc >= (-150. * fpm)) & (roc <= (150. * fpm)))

    cr = CR_alt * CR_roc * CR

    # -------------------------------------------------
    # phase SC[41]: alt >= FL100 , roc == 500
    SC_alt = np.array(alt > (100. * 100 * ft))
    SC_roc = np.array(roc == (500. * fpm))

    sc = SC_alt * SC_roc * SC

    # -------------------------------------------------
    # phase SD[42]: alt >= FL100 , roc == -500
    SD_alt = np.array(alt > (100. * 100 * ft))
    SD_roc = np.array(roc == (-500. * fpm))

    sd = SD_alt * SD_roc * SD

    # -------------------------------------------------
    # phase DE[5]: alt >= FL100, roc <= -150
    DE_alt = np.array(alt >= (100. * 100 * ft))
    DE_roc = np.array(roc <= (-150. * fpm))

    de = DE_alt * DE_roc * DE

    # -------------------------------------------------
    # phase AP[6]: 2000 <= alt <= 10000, roc <= 0
    AP_alt = np.array((alt >= (2000. * ft)) & (alt <= (10000. * ft)))
    AP_roc = np.array(roc <= (0 * fpm))

    ap = AP_alt * AP_roc * AP

    # -------------------------------------------------
    # phase LD[7]: alt <=2000, roc < 0
    LD_alt = np.array(alt <= (2000. * ft))
    LD_roc = np.array(roc < (0 * fpm))

    ld = LD_alt * LD_roc * LD

    # -------------------------------------------------
    # phase GD[8]: alt <= 1,
    GD_alt = np.array(alt <= (1. * ft))

    gd = GD_alt  * GD

    # -------------------------------------------------
    # combine all phases
    phase = np.maximum.reduce([to, ic, cl, cr, sc, sd, de, ap, ld])

    to2 = np.where(phase == TO)
    ic2 = np.where(phase == IC)
    cl2 = np.where(phase == CL)
    cr2 = np.where(phase == CR)
    sc2 = np.where(phase == SC)
    sd2 = np.where(phase == SD)
    de2 = np.where(phase == DE)
    ap2 = np.where(phase == AP)
    ld2 = np.where(phase == LD)
    # gd2 = np.where(phase == GD)

    # assign aircraft to their nominal bank angle per phase
    bank[to2] = bphase[0]
    bank[ic2] = bphase[1]
    bank[cl2] = bphase[2]
    bank[cr2] = bphase[3]
    bank[sc2] = bphase[3]
    bank[sd2] = bphase[3]
    bank[de2] = bphase[4]
    bank[ap2] = bphase[5]
    bank[ld2] = bphase[6]
    # bank[gd2] = bphase[7]

    # not turning aircraft do not have a bank angle
    noturn = np.array(swhdgsel) * 100
    bank   = np.minimum(noturn,bank)

    return (phase, bank)

#------------------------------------------------------------------------------
# ENERGY SHARE FACTOR
# (BADA User Manual 3.12, p.15)
#-----------------------------------------------------------------------------
def esf(alt, M, climb, descent, delspd, selmach,tISA= 1):

    # # overwrite delspd when geo descent
    # delspd = np.where(geoswitch, 0, delspd)

    # test for acceleration / deceleration
    cspd  = np.array((delspd <= 0.001) & (delspd >= -0.001))
    # accelerating or decelerating
    acc   = np.array(delspd > 0.001)
    dec   = np.array(delspd < -0.001)

    # tropopause
    abtp  = np.array(alt > 11000.0)
    beltp = np.logical_not(abtp)

    selcas = np.logical_not(selmach)

    # constant Mach/CAS
    # case a: constant MA above TP
    efa   = np.logical_and.reduce([cspd, selmach, abtp]) * 1

    # case b: constant MA below TP (at the moment just ISA: tISA = 1)
    # tISA = (self.temp-self.dtemp)/self.temp
    efb   = 1.0 / ((1.0 + ((gamma * R * beta) / (2.0 * g0)) * M**2 * tISA)) \
        * np.logical_and.reduce([cspd, selmach, beltp]) * 1

    # case c: constant CAS below TP (at the moment just ISA: tISA = 1)
    efc = 1.0 / (1.0 + (((gamma * R * beta) / (2.0 * g0)) * (M**2) * tISA) +
        ((1.0 + gamma1 * (M**2))**(-1.0 / (gamma - 1.0))) *
        (((1.0 + gamma1 * (M**2))**gamma2) - 1)) * \
        np.logical_and.reduce([cspd, selcas, beltp]) * 1

    #case d: constant CAS above TP
    efd = 1.0 / (1.0 + ((1.0 + gamma1 * (M**2) * tISA)**(-1.0 / (gamma - 1.0))) *
        (((1.0 + gamma1 * (M**2))**gamma2) - 1.0)) * \
        np.logical_and.reduce([cspd, selcas, abtp]) * 1

    #case e: acceleration in climb
    efe    = 0.3 * np.logical_and.reduce([acc, climb])

    #case f: deceleration in descent
    eff    = 0.3 * np.logical_and.reduce([dec, descent])

    #case g: deceleration in climb
    efg    = 1.7 * np.logical_and.reduce([dec, climb])

    #case h: acceleration in descent
    efh    = 1.7 * np.logical_and.reduce([acc, descent])

    # combine cases
    ef = np.maximum.reduce([efa, efb, efc, efd, efe, eff, efg, efh])

    # ESF of non-climbing/descending aircraft is zero which
    # leads to an error. Therefore, ESF for non-climbing aircraft is 1
    return np.maximum(ef, np.array(ef == 0) * 1)

#------------------------------------------------------------------------------
# CALCULATE LIMITS
#------------------------------------------------------------------------------
def calclimits(desspd, gs, to_spd, vmin, vmo, mmo, M, alt, hmaxact,
           desalt, desvs, maxthr, Thr, D, tas, mass, ESF, phase):

    # minimum CAS - below crossover (we do not check for minimum Mach)
    limspd      = np.where((desspd < vmin), vmin, -999.)

    # in traf, we will check for min and max spd, hence a flag is required
    limspd_flag = np.where((desspd < vmin), True, False)

    # maximum CAS: below crossover and above crossover
    limspd      = np.where((desspd > vmo), vmo, limspd )
    limspd_flag = np.where((desspd > vmo), True, limspd_flag)

    # maximum Mach
    limspd      = np.where((M > mmo), vmach2cas(mmo, alt), limspd)
    limspd_flag = np.where((M > mmo), True, limspd_flag)

    # remove non-needed limits
    limspd_flag = np.where((np.abs(desspd-limspd) <0.1), False, limspd_flag)
    limspd      = np.where((limspd_flag==False), -999.,limspd)

    # set altitude to max. possible altitude if alt>Hmax
    limalt = np.where((desalt>hmaxact), hmaxact -1.0, -999.)
    limalt_flag = np.where((desalt>hmaxact), True, False)

    # remove non-needed limits
    limalt = np.where((np.abs(desalt-hmaxact)<0.1), -999., limalt)
    limalt_flag = np.where((np.abs(desalt-hmaxact)<0.1), False, limalt_flag)

    # thrust and vertical speed
    Thr_corrected   = np.where((Thr > maxthr-1.0), maxthr-1., Thr)
    limvs = ((Thr_corrected - D) * tas) / (mass * g0)* ESF
    limvs_flag = True

    # aircraft can only take-off as soon as their speed is above v_rotate
    # True means that current speed is below rotation speed
    # limit vertical speed is thrust limited and thus should only be
    # applied for aircraft that are climbing
    limvs       = np.where((desvs > 0.) & (gs<to_spd) & (phase == 8), 0.0, limvs)
    limvs_flag  = np.where ((desvs > 0.) & (gs<to_spd) & (phase == 8), True, limvs_flag)


    # remove takeoff limit
    limvs      = np.where ((np.abs(to_spd - gs) < 0.1) & ((phase == 8) | (phase == 1)), -9999.,  limvs)
    limvs_flag = np.where((np.abs(to_spd - gs) < 0.1) & ((phase == 8) | (phase == 1)), True, limvs_flag)



    # remove non-needed limits
    Thr        = np.where((maxthr-Thr< 2.), -9999., Thr)
    limvs      = np.where((maxthr-Thr< 2.), -9999., limvs)
    limvs_flag = np.where((limvs< -999.), False, limvs_flag)


    return limspd, limspd_flag, limalt, limalt_flag, limvs, limvs_flag