""" iLabs Autopilot logic implementation"""
import numpy as np
from collections.abc import Collection
import bluesky as bs
from bluesky import stack
from bluesky.tools import geo
from bluesky.tools.misc import degto180
from bluesky.tools.aero import ft, nm, fpm, kts, vcas2tas, vcasormach2tas
from bluesky.traffic.autopilot import Autopilot, distaccel
from bluesky.core import timed_function
from bluesky.traffic.performance.iLP.iLP_ESTIMATOR import EEI
from bluesky.traffic.route import Route
from bluesky.traffic.performance.iLP.performance import PHASE
from .descent import Descent


def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name': "iLP_AP",
        'plugin_type': 'sim'
    }

    # Implement iLP_AP instead of Autopilot
    stack.stack("impl autopilot ilp_ap")

    return config

bs.settings.set_variable_defaults(scenario_path_SIDs='EHAM_SID')

class iLP_AP(Autopilot):
    """iLabs Autopilot implementation"""
    def __init__(self):
        super().__init__()

        # Standard self.steepness for descent
        self.steepness = 3000. * ft / (10. * nm)

        self.idxreached = []

        # From here, define object arrays
        with self.settrafarrays():
            # FMS directions
            self.trk = np.array([])
            self.spd = np.array([])
            self.tas = np.array([])
            self.alt = np.array([])
            self.vs = np.array([])

            # Speed Schedule
            self.spdschedule = np.array([])

            # VNAV variables
            self.swtoc = np.array([])           # ToC switch to switch on VNAV Top of Climb logic (default value True)
            self.swtod = np.array([])           # ToD switch to switch on VNAV Top of Descent logic (default value True)
            self.dist2vs = np.array([])         # distance from coming waypoint to TOD
            self.dist2accel = np.array([])      # Distance to go to acceleration(decelaration) for turn next waypoint [nm]
            self.swvnavvs = np.array([])        # whether to use given VS or not
            self.vnavvs = np.array([])          # vertical speed in VNAV
            # self.startdescent = np.array([])    # Start descent switch

            # LNAV variables
            self.qdr2wp = np.array([])          # Direction to waypoint from the last time passing was checked
                                                # to avoid 180 turns due to updated qdr shortly before passing wp
            self.dist2wp = np.array([])         # [m] Distance to active waypoint
            self.qdrturn = np.array([])         # qdr to next turn]
            self.dist2turn = np.array([])       # Distance to next turn [m]
            self.inturn = np.array([])          # If we're in a turn maneuver or not

            # Trffic navigation information
            self.orig = []                      # Four letter ICAO code of orgin airport
            self.dest = []                      # Four letter ICAO code of destination airport

            # Default values
            self.bankdef = np.array([])         # nominal bank angle, [radians]
            self.vsdef = np.array([])           # [m/s]default vertical speed of autopilot

            # Currently used roll/bank angle [rad]
            self.turnphi = np.array([])         # [rad] bank angle setting of autopilot

            # Route objects
            self.route = []

            # Takeoff Estimator information
            self.EEI = EEI()                    # Estimator class
            self.sd_spd = np.array([])          # Array containing performance levels for speed
            self.sd_roc = np.array([])          # Array containing performance levels for ROC
            self.calt = np.array([])            # Array containing cruise altitude
            self.cspd = np.array([])            # Array containing cruise speed
            self.TOsw = np.array([])            # Array containing take-off switches for the agents
            self.TOdf = np.array([], dtype = 'object')      # Array containing the climb dataframe objects
            self.caltdf = np.array([], dtype = 'object')    # Array containing the cruise altitude dataframe objects
            self.EEI_IAS = np.array([])         # Array with IAS for all aircraft agents with a TOsw
            self.EEI_ROC = np.array([])         # Array with ROC for all aircraft agents with a TOsw
            self.EEI_MACH = np.array([])        # Array with Mach for all aircraft agents with a TOsw
            self.EEI_COalt = np.array([])       # Array with crossover altitude for all aircraft agents with a TOsw
            self.EEI_SPD = np.array([])         # Array with Mach or IAS for all aircraft agents with a TOsw
            self.setspd = np.array([])          # This array is required for the event where somebody uses
                                                # the SPD command, and makes sure that this speed is selected.

            # Phase detection and recognition
            self.phase = np.array([])           # Array with current flight phase defined by performance

            #TODO: For now kept descent strategy from TU Delft

            # Descent Path switch
            self.geodescent = np.array([])

            #TODO: Descent variables from iLabs still need to be added (increase as little new variables as possible)

            # leg-based autoilot
            self.descenttype = np.array([])     # Array containing descent type, climb(-1), level(0), perf(1), geo(2)
            self.descentslope = np.array([])    # Array containing descent glide slope

            # gains for glideslope controller
            self.p_gs = np.array([])            # proportional gain for following glideslope

    def create(self, n=1):
        super().create(n)

        # FMS directions
        self.trk[-n:] = bs.traf.trk[-n:]
        self.tas[-n:] = bs.traf.tas[-n:]
        self.alt[-n:] = bs.traf.alt[-n:]
        self.vs[-n:] = -999.

        # Default ToC/ToD logic on
        self.swtoc[-n:] = True                 # Default top of climb switch is False
        self.swtod[-n:] = True                 # Default top of descent switch is False

        # VNAV Variables
        self.dist2vs[-n:] = -999.
        self.dist2accel[-n:] = -999.            # Distance to go to (de)/acceleration for turn next waypoint [nm]
        # self.startdescent[-n:] = False          # Default start descent as False

        # LNAV variables
        self.qdr2wp[-n:] = -999.                # Direction to waypoint from the last time passing was checked
        self.dist2wp[-n:]  = -999.              # Distance to go to next waypoint [nm]

        # Traffic performance data (temporarily default values)
        self.vsdef[-n:] = 1500 * fpm            # default vertical speed of autopilot
        self.bankdef[-n:] = np.radians(25.)     # default bank angle

        # Route objects
        for ridx, acid in enumerate(bs.traf.id[-n:]):
            self.route[ridx - n] = Route(acid)

        # Takeoff Estimator information
        self.sd_spd[-n:] = 0
        self.sd_roc[-n:] = 0
        self.calt[-n:] = None
        self.cspd[-n:] = None
        self.TOsw[-n:] = False
        self.TOdf[-n:] = False
        self.caltdf[-n:] = False
        self.EEI_IAS[-n:] = 0
        self.EEI_ROC[-n:] = 0
        self.EEI_MACH[-n:] = 0
        self.EEI_COalt[-n:] = 0
        self.EEI_SPD[-n:] = 0
        self.setspd[-n:] = 0

        # leg-base descent autopilot
        self.descentslope[-n:] = 0
        self.descenttype[-n:] = 0

        # Descent path swtich
        self.geodescent[-n:] = False

        # gains for glideslope controller
        self.p_gs[-n] = -.02                # default value of P-gain

    def update(self):
        """
        VNAV: follow trajectory, set VS according to performance model, decide ToD
        LNAV: follow route, follow commands

        Returns:
            VS, TAS to fly, hdg
        """
        # Vector to next WP
        qdr, distinnm = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    bs.traf.actwp.lat, bs.traf.actwp.lon)  # [deg][nm])
        self.qdr2wp = qdr
        self.dist2wp = distinnm*nm  # Conversion to meters

        # Check possible waypoint shift. Note: qdr, dist2wp will be updated accordingly in case of wp switch
        self.wppassingcheck(qdr, self.dist2wp)  # Updates self.qdr2wp when necessary

        # Phase Detection
        self.phase = bs.traf.perf.phase

        # In case TOD was not used, use current given CAS and ALT as cruise altitude and speed
        self.cspd = np.where(self.TOsw, self.cspd, bs.traf.cas)
        self.calt = np.where(self.TOsw, self.calt, bs.traf.alt)

        # which a/c reached their ToD command altitude/ cleared altitude
        reached_tod = self.dist2wp < self.dist2vs
        reached_alt = bs.traf.alt - bs.traf.selalt < 10
        reached_climb = np.where((self.phase < PHASE['CR']) & (self.phase >= PHASE["TO"]), True, False)


        # ==============================================================================================================
        # VNAV autopilot logic
        # ==============================================================================================================

        # # start descent if swvnav == True, ToD is reached, at or after cruise phase, reached altitude is false
        # self.swtod = np.where(bs.traf.swvnav * reached_tod * (self.phase >= PHASE['CR']) *
        #                       ~reached_alt, True, self.swtod)
        #
        # # start climb if PHASE is before CR
        # self.swtoc = np.where(bs.traf.swvnav * reached_climb, True, False)
        #
        # # set selected altitude for climb
        # bs.traf.selalt = np.where(self.swtod, bs.traf.actwp.nextaltco, bs.traf.selalt)
        #
        # # set selected altitude for descent
        # bs.traf.selalt = np.where(self.swtod, bs.traf.actwp.nextaltco, bs.traf.selalt)
        #
        # # first find which type of descent phase each a/c is in
        # idx_climb = np.where(self.descenttype == -1)[0]
        # idx_level = np.where(self.descenttype == 0)[0]
        # idx_perf = np.where(self.descenttype == 1)[0]
        # idx_geo = np.where(self.descenttype == 2)[0]
        # idx_rate = np.where(self.descenttype == 3)[0]
        #
        # # then determine vertical speed
        # # _____________________________________________Level flight_____________________________________________________
        # self.vs[idx_level] = np.zeros(len(bs.traf.id))[idx_level]
        #
        # # __________________________________________Performance Descent_________________________________________________
        # # set vs at -999 m/s, this value will be limited by the performance model
        # self.vs[idx_perf] = np.where(self.swtod, -999, np.array([0]))[idx_perf]
        #
        # # ___________________________________________Geometric descent__________________________________________________
        # # find difference between desired altitude on glideslope and current altitude
        # #   used for glideslope tracking with P-controller
        # des_alt = np.tan(np.radians(self.descentslope)) * self.dist2wp + bs.traf.actwp.nextaltco
        # d_alt_geo = des_alt - bs.traf.alt
        # self.vs[idx_geo] = np.where(self.swtod,
        #                             np.radians(self.descentslope) * bs.traf.gs + d_alt_geo * self.p_gs,
        #                             np.array([0]))[idx_geo]
        #
        # # __________________________________________Fixed rate descent__________________________________________________
        # self.vs[idx_rate] = bs.traf.selvs[idx_rate]
        #
        # #_________________________________________________Climb_________________________________________________________
        # self.vs[idx_climb] = np.full(len(bs.traf.id),999)[idx_climb]
        #
        # # geometric descent is used by performance model to decide on flight strategy
        # bs.traf.actwp.vs = np.where(np.isin(self.descenttype, [2,3]), self.steepness, -999)
        # selvs = np.where(abs(bs.traf.selvs) > 0.1, bs.traf.selvs, self.vsdef)       # given in m/s


        # TU Delft Bleusky Version
        startdescorclimb = (bs.traf.actwp.nextaltco >= -0.1) * \
                           np.logical_or((bs.traf.alt > bs.traf.actwp.nextaltco) * \
                                         np.logical_or((self.dist2wp < self.dist2vs + bs.traf.actwp.turndist),
                                                       (np.logical_not(self.swtod))),
                                         bs.traf.alt < bs.traf.actwp.nextaltco)


        # If not lnav:Climb/descend if doing so before lnav/vnav was switched off
        #    (because there are no more waypoints). This is needed
        #    to continue descending when you get into a conflict
        #    while descending to the destination (the last waypoint)
        #    Use 0.1 nm (185.2 m) circle in case turndist might be zero
        self.swvnavvs = bs.traf.swvnav * np.where(bs.traf.swlnav, startdescorclimb,
                                                  self.dist2wp <= np.maximum(0.1 * nm, bs.traf.actwp.turndist))

        # Recalculate V/S based on current altitude and distance to next alt constraint
        # How much time do we have before we need to descend?
        # Now done in ComputeVNAV
        # See ComputeVNAV for bs.traf.actwp.vs calculation

        self.vnavvs = np.where(self.swvnavvs, bs.traf.actwp.vs, self.vnavvs)
        # was: self.vnavvs  = np.where(self.swvnavvs, self.steepness * bs.traf.gs, self.vnavvs)

        # self.vs = np.where(self.swvnavvs, self.vnavvs, self.vsdef * bs.traf.limvs_flag)
        # for VNAV use fixed V/S and change start of descent
        selvs = np.where(abs(bs.traf.selvs) > 0.1, bs.traf.selvs, self.vsdef)  # m/s
        self.vs = np.where(self.swvnavvs, self.vnavvs, selvs)
        self.alt = np.where(self.swvnavvs, bs.traf.actwp.nextaltco, bs.traf.selalt)

        # When descending or climbing in VNAV also update altitude command of select/hold mode
        bs.traf.selalt = np.where(self.swvnavvs, bs.traf.actwp.nextaltco, bs.traf.selalt)



        # ==============================================================================================================

        # ==============================================================================================================
        # LNAV autopilot logic
        # ==============================================================================================================

        # LNAV commanded track angle
        self.trk = np.where(bs.traf.swlnav, self.qdr2wp, self.trk)

        # FMS speed guidance: anticipate accel/decel distance for next leg or turn

        # Calculate actual distance it takes to decelerate/accelerate based on two cases: turning speed (decel)

        # Normally next leg speed (actwp.spd) but in case we fly turns with a specified turn speed
        # use the turn speed

        # Is turn speed specified and are we not already slow enough? We only decelerate for turns, not accel.
        turntas = np.where(bs.traf.actwp.turnspd > 0.0, vcas2tas(bs.traf.actwp.turnspd, bs.traf.alt),
                           -1.0 + 0. * bs.traf.tas)

        # Switch is now whether the aircraft has any turn waypoints
        swturnspd = bs.traf.actwp.nextturnidx >= 0

        # t = (v1-v0)/a ; x = v0*t+1/2*a*t*t => dx = (v1*v1-v0*v0)/ (2a)
        dxturnspdchg = distaccel(turntas, bs.traf.tas, bs.traf.perf.axmax)

        # Decelerate or accelerate for next required speed because of speed constraint or RTA speed
        # Note that because nextspd comes from the stack, and can be either a mach number or
        # a calibrated airspeed, it can only be converted from Mach / CAS [kts] to TAS [m/s]
        # once the altitude is known.
        nexttas = vcasormach2tas(bs.traf.actwp.nextspd, bs.traf.alt)
        dxspdconchg = distaccel(bs.traf.tas, nexttas, bs.traf.perf.axmax)

        qdrturn, dist2turn = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                        bs.traf.actwp.nextturnlat, bs.traf.actwp.nextturnlon)

        self.qdrturn = qdrturn
        dist2turn = dist2turn * nm

        # Update speed schedule after cruise phase
        #if self.phase > PHASE['CR']:
        #    self.descent_speedschedule(bs.traf.alt)

        # Determine if the speed schedule should be used
        # if own speed is slower than the speed schedule own speed is used
        use_speedschedule = (self.spdschedule < bs.traf.actwp.spdcon) * (self.phase > PHASE['CR'])
        self.spdschedule = np.where(vcasormach2tas(self.spdschedule, bs.traf.alt) >
                                    (1.01 * vcasormach2tas(bs.traf.selspd, bs.traf.alt)) * (bs.traf.selspd > 1),
                                    bs.traf.selspd, self.spdschedule)


        # Where we don't have a turn waypoint, as in turn idx is negative, then put distance
        # as Earth circumference.
        self.dist2turn = np.where(bs.traf.actwp.nextturnidx >= 0, dist2turn, 40075000)

        # Check also whether VNAVSPD is on, if not, SPD SEL has override for next leg
        # and same for turn logic
        usenextspdcon = (self.dist2wp < dxspdconchg) * (bs.traf.actwp.nextspd > -990.) * \
                        bs.traf.swvnavspd * bs.traf.swvnav * bs.traf.swlnav


        useturnspd = np.logical_or(bs.traf.actwp.turntonextwp,
                                   (self.dist2turn < (dxturnspdchg * 1.1 + np.maximum(bs.traf.actwp.turndist,
                                                                                      bs.traf.actwp.nextturnrad)))) * \
                     swturnspd * bs.traf.swvnavspd * bs.traf.swvnav * bs.traf.swlnav

        # Hold turn mode can only be switched on here, cannot be switched off here (happeps upon passing wp)
        bs.traf.actwp.turntonextwp = bs.traf.swlnav * np.logical_or(bs.traf.actwp.turntonextwp, useturnspd)

        # Which CAS/Mach do we have to keep? VNAV, last turn or next turn?
        oncurrentleg = (abs(degto180(bs.traf.trk - qdr)) < 2.0) # [deg]
        inoldturn    = (bs.traf.actwp.oldturnspd > 0.) * np.logical_not(oncurrentleg)

        # Avoid using old turning speeds when turning of this leg to the next leg
        # by disabling (old) turningspd when on leg
        bs.traf.actwp.oldturnspd = np.where(oncurrentleg*(bs.traf.actwp.oldturnspd>0.), -998.,
                                            bs.traf.actwp.oldturnspd)

        # turnfromlastwp can only be switched off here, not on (latter happens upon passing wp)
        bs.traf.actwp.turnfromlastwp = np.logical_and(bs.traf.actwp.turnfromlastwp, inoldturn)

        # Select speed: turn sped, next speed constraint, or current speed constraint
        bs.traf.selspd = np.where(useturnspd,bs.traf.actwp.nextturnspd,
                                  np.where(usenextspdcon, bs.traf.actwp.nextspd,
                                           np.where((bs.traf.actwp.spdcon>=0)*bs.traf.swvnavspd,
                                                   np.where(use_speedschedule,self.spdschedule,bs.traf.actwp.spdcon),
                                                    np.where(bs.traf.swvnavspd & (self.phase > PHASE['CR']),
                                                                        self.spdschedule,bs.traf.selspd))))

        # Temporary override speed when still in old turn
        bs.traf.selspd = np.where(np.logical_and(inoldturn*(bs.traf.actwp.oldturnspd>0.)*bs.traf.swvnavspd*bs.traf.swvnav*bs.traf.swlnav,
                                    np.logical_not(useturnspd)),
                                  bs.traf.actwp.oldturnspd,bs.traf.selspd)

        # Before updating inturn, save the old inturn
        oldinturn = self.inturn

        self.inturn = np.logical_or(useturnspd, inoldturn)

        # Turn was exited this time step
        justexitedturn = np.logical_and(oldinturn, np.logical_not(self.inturn))

        # Apply the cruise speed if the past or next waypoint doesn't have a speed constraint
        # and if there is actually a cruise speed to apply
        usecruisespd = np.logical_and.reduce((self.cspd > 0,
                                              bs.traf.actwp.spd < 0,
                                              np.logical_not(usenextspdcon),
                                              justexitedturn))

        bs.traf.selspd = np.where(usecruisespd, self.cspd, bs.traf.selspd)


        # ==============================================================================================================


        # ==============================================================================================================
        # The following lines consider with the implementation of the EEI function required for the realistic
        # take-off and climb performance.
        # ==============================================================================================================
        if np.any(self.phase < PHASE['CR']) and np.any(self.TOsw):
            # Timed function which every dt seconds aims to update the self.EEI_IAS, self.EEI_ROC, self.EEI_MACH and
            # self.EEI_COalt arrays.
            self.ESTspeeds()
            self.EEI_IAS, self.EEI_ROC, self.EEI_MACH, self.EEI_COalt = (
                self.EEI.IAS_ROC(self.TOdf, bs.traf.alt, bs.traf.cas, self.sd_spd, self.sd_roc, self.calt))

            # Check whether the agent has a speed constraint. If yes, use the minimum of the speed constraint and
            # estimation in the IAS array.
            spdcon = np.logical_not(bs.traf.actwp.nextspd <= 0)
            self.EEI_IAS = np.where(spdcon, np.minimum(bs.traf.actwp.nextspd, self.EEI_IAS), self.EEI_IAS)

            # Check whether the agent has a speed constraint. If yes, yse the minimum of the speed constraint and
            # estimation in the IAS array
            spdcon = np.logical_not(self.setspd <= 0)
            self.EEI_IAS = np.where(spdcon, np.minimum(bs.traf.actwp.nextspd, self.EEI_IAS),self.EEI_IAS)

            # Check for CAS/MACH crossover altitude (COL)
            self.EEI_SPD = np.where(bs.traf.alt >= (self.EEI_COalt * 100 * ft), self.EEI_MACH, self.EEI_IAS)

            # Selected speed is equal to the estimation in the array.
            bs.traf.selspd = np.where(self.TOsw, self.EEI_SPD, bs.traf.selspd)

            # These switches are required to prohibit aircraft from taking off as soon as they have a velocity greater
            # than 0. Select either 0 or the estimated ROC.
            sw_vs_restr = abs(self.alt - bs.traf.alt) > 10
            bs.traf.vs = np.where(self.TOsw * ~reached_tod * self.phase == 1,
                                  np.where(sw_vs_restr,
                                           np.where(self.TOsw, self.EEI_ROC, selvs),0),
                                  bs.traf.vs)

            # Update vertical speed
            self.vs = np.where(self.TOsw, self.EEI_ROC, self.vs)

            # Set cruise altitude after last active waypoint in climb
            bs.traf.selalt = np.where(abs(bs.traf.alt - bs.traf.selalt) < 10, self.calt, bs.traf.selalt)

        # ==============================================================================================================

        # Check if cruise phase has been reached, if so set speed to cruise speed
        bs.traf.selspd = np.where(self.phase == PHASE['CR'], self.cspd, bs.traf.selspd)

        # Below crossover altitude: CAS=const, above crossover altitude: Mach = const
        self.tas = vcasormach2tas(bs.traf.selspd, bs.traf.alt)

        if self.phase >= PHASE['CR'] and self.phase != PHASE['GD']:
            for i in self.idxreached:
                bs.traf.swvnav[i] = True
                bs.traf.swvnavspd[i] = True
                self.TOsw[i] = False


    @timed_function(dt=3)
    def ESTspeeds(self):
        '''
        Function: Update the speed and ROC in the self.EEI_IAS, self.EEI_ROC, self.EEI_MACH, and self.EEI_COalt arrays.
        This function is only executed every 3 seconds (simtime) to decrease the computations required.

        Created by: Lars Dijkstra
        Date: somewhere early November 2022

        Edited by: Amre Dardor
        Date: July 2023
        '''

        self.EEI_IAS, self.EEI_ROC, self.EEI_MACH, self.EEI_COalt = (
            self.EEI.IAS_ROC(self.TOdf, bs.traf.alt, bs.traf.cas, self.sd_spd, self.sd_roc, self.calt))


    @stack.command(name='TOD', annotations="acid,txt,txt,alt,spd,float,float")
    def TODcmd(self, idx: 'acid', SID: 'SID' = None, dest: 'dest' = None, calt: 'alt' = 0, cspd: 'spd' = 0,
               sd_spd: 'sd' = 0, sd_roc: 'sd' = 0):
        """
        Function: Set the take-off switch for aircraft agent X to True or False. In case the take-off switch
        is False, the take-off switch is set to True and the aircraft will select the speed as obtained from the
        estimations. If the switch was True, the take-off switch will be set to False for this aircraft agent
        and its speed and ROC will no longer be prescribed by the estimation class.

        This function ensures that the highest degree of accuracy in take-off performance is obtained from the
        data-set file given the inputs.

        Args:
            acid:   Aircraft ID
            SID:    Standard Instrument Departure (use 5-letter code e.g. ARN4E)
            dest:   Destination airport
            calt:   Cruise altitude for the aircraft
            cspd:   Cruise IAS for the aircraft
            sd_spd: Performance Level for speed. +ve means higher speed
            d_roc: Performance Level for climb. +ve means faster climb
        Returns:
            None

        Created by: Lars Dijkstra
        Date: somewhere early October 2022

        Edited by: Arham Elahi
        Date: July-August 2024

        Edited by: Albert Chou
        Date: May-June 2025
        """
        if not self.TOsw[idx]:
            self.TOsw[idx] = True
            bs.traf.swvnav[idx] = False
            id = bs.traf.id[idx]
            AC_type = bs.traf.type[idx]
            self.sd_spd[idx] = sd_spd
            self.sd_roc[idx] = sd_roc


            if SID == "NULLSID":
                print("Ensure Waypoints for travelling are given, otherwise aircraft will not move")

            if "/" in dest:
                airport, _ = dest.split("/")
            else:
                airport = dest

            self.TOdf[idx], _, SID = self.EEI.select(id, AC_type, SID=SID, airline=id[:3],
                                                                      DEST=airport)
            self.caltdf[idx], _, _ = self.EEI.select(id, AC_type, SID=None, airline=id[:3], DEST=airport)

            calt_FL = int(calt / (ft * 100))

            if calt_FL == 0:
                calt_FL = int(self.caltdf[idx]['mode_calt'].max())
                calt = calt_FL * (ft * 100)
                bs.scr.echo(f"Cruise altitude not provided, FL{int(calt_FL)} used instead")

            if calt_FL > np.nanmin([self.caltdf[idx]['max_calt'].max(), self.TOdf[idx]['ALT'].max()]):
                calt_FL = int(np.nanmin([self.caltdf[idx]['max_calt'].max(), self.TOdf[idx]['ALT'].max()]))
                calt = calt_FL * (ft * 100)
                bs.scr.echo(f"Cruise altitude is too high, FL{int(calt_FL)} used instead")

            if cspd == 0:
                cruise_row = self.TOdf[idx].loc[self.TOdf[idx]['ALT'] == int(calt_FL)].iloc[:1]
                if sd_spd == 0:
                    cspd = cruise_row.iloc[0]['IAS'] * kts
                else:
                    cspd = (cruise_row.iloc[0]['IAS'] + sd_spd * cruise_row.iloc[0]["IAS_sd"]) * kts

            self.calt[idx] = calt
            self.cspd[idx] = cspd

            # Setting a default SID route in case no SID is given, does not influence performance
            if SID is None:
                SID = "AND2S"

            # Add SID waypoints to routes
            string = bs.settings.scenario_path_SIDs + "/" + SID
            stack.simstack.pcall(string, id)
            self.setdest(acidx=idx, wpname=dest)



    def descent_speedschedule(self, alt):
        # TODO: Make this compatible with OpenAP
        '''
        Function: Calculate the speeds corresponding to the speed schedule of BADA manual 3.10
        For either jet/turboprop or piston aircraft
        Only for descent currently!

        Created by: Winand Mathoera
        Date: 26/09/2022
        '''

        alt = alt / ft

        # Correction for minimum landing speed
        corr = np.sqrt(bs.traf.perf.mass / bs.traf.perf.mref)

        # Calculate the speeds for each altitude segment based on either
        # the minimum landing speed or the standard descent speed
        l1 = corr * bs.traf.perf.vmld / kts + 5
        l2 = corr * bs.traf.perf.vmld / kts + 10
        l3 = corr * bs.traf.perf.vmld / kts + 20
        l4 = corr * bs.traf.perf.vmld / kts + 50
        l5 = np.where(bs.traf.perf.casdes / kts <= 220, bs.traf.perf.casdes / kts, 220)
        l6 = np.where(bs.traf.perf.casdes2 / kts <= 250, bs.traf.perf.casdes2 / kts, 250)
        l7 = 280
        l8 = np.where(bs.traf.perf.mmax > 7000, 0.78, 0.82)

        # Piston aircraft speeds
        # TODO: Check the Vdes_i value
        l9 = corr * bs.traf.perf.vmld / kts + 0
        l10 = corr * bs.traf.perf.vmld / kts + 0
        l11 = corr * bs.traf.perf.vmld / kts + 0
        l12 = bs.traf.perf.casdes
        l13 = bs.traf.perf.casdes2
        l14 = bs.traf.perf.mades

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
                 np.logical_and(alt > 9999, alt <= bs.traf.perf.hpdes / 0.3048) * l13 * kts + \
                 np.where(alt > bs.traf.perf.hpdes / 0.3048, 1, 0) * l14

        self.spdschedule = np.where(bs.traf.perf.piston == 1, spds_p, spds)
