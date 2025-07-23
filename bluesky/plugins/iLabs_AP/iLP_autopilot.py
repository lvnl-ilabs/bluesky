""" iLabs Autopilot logic implementation"""
import numpy as np
from collections.abc import Collection
import bluesky as bs
from bluesky import stack
from bluesky.tools import geo
from bluesky.tools.misc import degto180
from bluesky.tools.position import txt2pos
from bluesky.tools.aero import ft, nm, fpm, kts, vcas2tas, vcasormach2tas, cas2tas, g0
from bluesky.traffic.autopilot import Autopilot, distaccel
from bluesky.core import timed_function
from bluesky.traffic.performance.iLP.iLP_ESTIMATOR import EEI
from bluesky.traffic.route import Route
from bluesky.traffic.performance.iLP.performance import PHASE
from .descent import Descent
from .vnav_ap import VNAV
from bluesky.traffic.performance.perfbase import PerfBase


performance_model = PerfBase.selected().__name__.lower()

def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name': "iLP_AP",
        'plugin_type': 'sim'
    }

    # Construct once
    _ = iLP_AP()

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
            self.swtoc = np.array([])           # ToC switch to switch on VNAV Top of Climb logic (default value False)
            self.swtod = np.array([])           # ToD switch to switch on VNAV Top of Descent logic (default value False)
            self.dist2vs = np.array([])         # distance from coming waypoint to TOD
            self.dist2accel = np.array([])      # Distance to go to acceleration(decelaration) for turn next waypoint [nm]
            self.swvnavvs = np.array([])        # whether to use given VS or not
            self.vnavvs = np.array([])          # vertical speed in VNAV
            self.startdescent = np.array([])    # Start descent switch
            self.vnav = VNAV()                  # Initiating VNAV_AP
            self.selvs = np.array([])           # Array with selected VS for each a/c
            self.cleared_alt = np.array([])


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

            # leg-based autoilot
            self.descenttype = np.array([])     # Array containing descent type, climb(-1), level(0), perf(1), geo(2)
            self.descentslope = np.array([])    # Array containing descent glide slope
            self.geodescent = np.array([])      # Array containing geo descent path switch
            self.descentalt = np.array([])      # Array containing descent altitude
            self.descentleg = np.array([])      # Array containing descent leg distance

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
        self.swtoc[-n:] = False                 # Default top of climb switch is False
        self.swtod[-n:] = False                # Default top of descent switch is False

        # VNAV Variables
        self.dist2vs[-n:] = -999.
        self.dist2accel[-n:] = -999.            # Distance to go to (de)/acceleration for turn next waypoint [nm]
        self.startdescent[-n:] = False          # Default start descent as False
        self.selvs[-n:] = 0                     # Default selected vertical speed to be level flight 0 [fpm]

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
        self.descentalt[-n:] = 0
        self.descentleg[-n:] = 0

        # Descent path swtich
        self.geodescent[-n:] = False

        # Cleared Altitude Information
        self.cleared_alt[-n:] = 0

        # gains for glideslope controller
        self.p_gs[-n] = -.02                # default value of P-gain

    def wppassingcheck(self, qdr, dist):
        """
        The actwp is the interface between the list of waypoint data in the route object and the autopilot guidance
        when LNAV is on (heading) and optionally VNAV is on (spd & altitude)

        actwp data contains traffic arrays, to allow vectorizing the guidance logic.

        Waypoint switching (just like the adding, deletion in route) are event driven commands and
        therefore not vectorized as they occur rarely compared to the guidance.

        wppassingcheck contains the waypoint switching function:
        - Check which aircraft i have reached their active waypoint
        - Reached function return list of indices where reached logic is True
        - Get the waypoint data to the actwp (active waypoint data)
        - Shift waypoint (last,next etc.) data for aircraft i where necessary
        - Shift and maintain data (see last- and next- prefix in varubale name) e.g. to continue a special turn
        """

        # Get list of indices of aircraft which have reached their active waypoint
        # This vectorized function checks the passing of the waypoint using a.o. the current turn radius
        self.idxreached = bs.traf.actwp.reached(qdr, dist)

        for i in self.idxreached:
            # Save current wp speed for use on next leg when we pass this waypoint
            # VNAV speeds are always FROM-speeds, so we accelerate/decellerate at the waypoint
            # where this speed is specified, so we need to save it for use now
            # before getting the new data for the next waypoint

            # Get speed for next leg from the waypoint we pass now and set as active spd
            bs.traf.actwp.spd[i] = bs.traf.actwp.nextspd[i]
            bs.traf.actwp.spdcon[i] = bs.traf.actwp.nextspd[i]

            # Execute stack commands for the still active waypoint, which we pass now
            self.route[i].runactwpstack()

            # Get next wp, if there still is one
            if not bs.traf.actwp.swlastwp[i]:
                lat, lon, alt, bs.traf.actwp.nextspd[i], \
                    bs.traf.actwp.xtoalt[i], toalt, \
                    bs.traf.actwp.xtorta[i], bs.traf.actwp.torta[i], \
                    lnavon, flyby, flyturn, turnrad, turnspd, turnhdgr, turnbank, \
                    bs.traf.actwp.next_qdr[i], bs.traf.actwp.swlastwp[i] = \
                    self.route[i].getnextwp()  # [m] note: xtoalt,nextaltco are in meters

                bs.traf.actwp.nextturnlat[i], bs.traf.actwp.nextturnlon[i], \
                    bs.traf.actwp.nextturnspd[i], bs.traf.actwp.nextturnrad[i], \
                    bs.traf.actwp.nextturnhdgr[i], bs.traf.actwp.nextturnidx[i] = \
                    self.route[i].getnextturnwp()

            else:
                # Prevent trying to activate the next waypoint when it was already the last waypoint
                # In case of end of route/no more waypoints: switch off LNAV using the lnavon
                bs.traf.swlnav[i] = False
                bs.traf.swvnav[i] = False
                bs.traf.swvnavspd[i] = False
                continue  # Go to next a/c which reached its active waypoint

            # Special turns: specified by one or two of the following variables:
            # Turn speed, turn radius, turn bank angle, turn rate
            if flyturn:
                # First situation, if turn rate is specified
                if turnspd<=0.:
                    turnspd = bs.traf.tas[i]

                # Heading rate overrides turnrad
                if turnhdgr>0:
                    turnrad = bs.traf.tas[i]*360./(2*np.pi*turnhdgr)

                # Use last turn radius for bank angle in current turn
                if bs.traf.actwp.turnrad[i] > 0.:
                    self.turnphi[i] = np.arctan(bs.traf.actwp.turnspd[i]*bs.traf.actwp.turnspd[i]/ \
                                           (bs.traf.actwp.turnrad[i]*g0)) # [rad]
                else:
                    self.turnphi[i] = 0.0  # [rad] or leave untouched???

            else:
                self.turnphi[i] = 0.0  #[rad] or leave untouched???

            # Check LNAV switch returned by getnextwp
            # Switch off LNAV if it failed to get next wpdata
            if not lnavon and bs.traf.swlnav[i]:
                bs.traf.swlnav[i] = False
                # Last wp: copy last wp values for alt and speed in autopilot
                if bs.traf.swvnavspd[i] and bs.traf.actwp.nextspd[i] >= 0.0:
                    bs.traf.selspd[i] = bs.traf.actwp.nextspd[i]

            # In case of no LNAV, do not allow VNAV mode to be active
            bs.traf.swvnav[i] = bs.traf.swvnav[i] and bs.traf.swlnav[i]

            bs.traf.actwp.lat[i] = lat  # [deg]
            bs.traf.actwp.lon[i] = lon  # [deg]
            # 1.0 in case of fly by, else fly over
            bs.traf.actwp.flyby[i] = int(flyby)

            # Update qdr and turndist for this new waypoint
            qdr[i], distnmi = geo.qdrdist(bs.traf.lat[i], bs.traf.lon[i],
                                          bs.traf.actwp.lat[i], bs.traf.actwp.lon[i])

            # dist[i] = distnmi * nm
            self.dist2wp[i] = distnmi * nm

            bs.traf.actwp.curlegdir[i] = qdr[i]
            bs.traf.actwp.curleglen[i] = self.dist2wp[i]

            # User has entered an altitude for the new waypoint
            if alt >= -0.01:  # positive alt on this waypoint means altitude constraint
                bs.traf.actwp.nextaltco[i] = alt  # [m]
                bs.traf.actwp.xtoalt[i] = 0.0
            else:
                bs.traf.actwp.nextaltco[i] = toalt  # [m]

            # VNAV spd mode: use speed of this waypoint as commanded speed
            # while passing waypoint and save next speed for passing next wp
            # Speed is now from speed! Next speed is ready in wpdata
            if bs.traf.swvnavspd[i] and bs.traf.actwp.spd[i] >= 0.0:
                bs.traf.selspd[i] = bs.traf.actwp.spd[i]

            # Update turndist, is there a next leg direction or not?
            if bs.traf.actwp.next_qdr[i] < -900.:
                local_next_qdr = qdr[i]
            else:
                local_next_qdr = bs.traf.actwp.next_qdr[i]

            # Get flyturn switches and data
            bs.traf.actwp.flyturn[i] = flyturn
            bs.traf.actwp.turnrad[i] = turnrad
            bs.traf.actwp.turnspd[i] = turnspd
            bs.traf.actwp.turnhdgr[i] = turnhdgr
            self.turnphi[i] = np.deg2rad(turnbank)

            # Pass on whether currently flyturn mode:
            # at beginning of leg,c copy tonextwp to lastwp
            # set next turn False
            bs.traf.actwp.turnfromlastwp[i] = bs.traf.actwp.turntonextwp[i]
            bs.traf.actwp.turntonextwp[i] = False

            # Keep both turning speeds: turn to leg and turn from leg
            bs.traf.actwp.oldturnspd[i] = bs.traf.actwp.turnspd[i]  # old turnspd, turning by this waypoint

            if bs.traf.actwp.flyturn[i]:
                bs.traf.actwp.turnspd[i] = turnspd  # new turnspd, turning by next waypoint
            else:
                bs.traf.actwp.turnspd[i] = -990.

            # Reduce turn dist for reduced turnspd
            if bs.traf.actwp.flyturn[i] and bs.traf.actwp.turnrad[i] < 0.0 and bs.traf.actwp.turnspd[i] >= 0.:
                turntas = cas2tas(bs.traf.actwp.turnspd[i], bs.traf.alt[i])
                bs.traf.actwp.turndist[i] = bs.traf.actwp.turndist[i] * turntas * turntas / (
                            bs.traf.tas[i] * bs.traf.tas[i])

        # End of reached-loop: the per waypoint i switching loop

        # Update qdr2wp with up-to-date qdr, now that we have checked passing w
        self.qdr2wp = qdr % 360.

        # Continuous guidance when speed constraint on active leg is in update-method

        # If still an RTA in the route and currently no speed constraint
        for iac in np.where((bs.traf.actwp.torta > -99.) * (bs.traf.actwp.spdcon < 0.0))[0]:
            iwp = bs.traf.ap.route[iac].iactwp
            if bs.traf.ap.route[iac].wprta[iwp] > -99.:

                # For all a/c flying to an RTA waypoint, recalculate speed more often
                dist2go4rta = geo.kwikdist(bs.traf.lat[iac], bs.traf.lon[iac],
                                           bs.traf.actwp.lat[iac], bs.traf.actwp.lon[iac]) * nm \
                              + bs.traf.ap.route[iac].wpxtorta[iwp]  # last term zero for active wp rta

                # Set bs.traf.actwp.spd to rta speed, if necessary
                self.setspeedforRTA(iac, bs.traf.actwp.torta[iac], dist2go4rta)

                # If VNAV speed is on (by default coupled to VNAV), use it for speed guidance
                if bs.traf.swvnavspd[iac] and bs.traf.actwp.spd[iac] >= 0.0:
                    bs.traf.selspd[iac] = bs.traf.actwp.spd[iac]

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

        # # Run VNAV logic for each aircraft after waypoint shift
        # for i in range(len(bs.traf.id)):
        #     self.vnav.update(self.route, i, self.idxreached)


        # Phase Detection
        self.phase = bs.traf.perf.phase

        # In case TOD was not used, use current given CAS and ALT as cruise altitude and speed, set ToC switch to True
        self.cspd = np.where(self.TOsw, self.cspd, bs.traf.cas)
        self.calt = np.where(self.TOsw, self.calt, bs.traf.alt)
        self.swtoc = np.where(self.TOsw, False, True)

        # ==============================================================================================================
        # VNAV autopilot logic
        # ==============================================================================================================

        # which a/c reached their ToD command altitude/ cleared altitude and ToC command
        reached_tod = self.dist2wp < self.dist2vs
        reached_alt = bs.traf.alt - bs.traf.selalt < 10
        reached_cleared_alt = bs.traf.alt - self.cleared_alt < 10

        # start descent if swvnav == True, ToD is reached, at or after cruise phase
        # stop descent if selalt reached or cleared alt reached
        self.startdescent = np.where(bs.traf.swvnav * reached_tod * self.phase >= PHASE["CR"], True, self.startdescent)
        self.startdescent = np.where(reached_alt, False, self.startdescent)
        self.startdescent = np.where(reached_cleared_alt, False, self.startdescent)

        # set ToC switch to True if Cruise_alt - Alt is lower than 5m
        self.swtoc = np.where(self.calt - bs.traf.alt < 10, True, False)

        # set selected altitude for climb
        bs.traf.selalt = np.where(bs.traf.swvnav * self.swtoc, bs.traf.actwp.nextaltco, bs.traf.selalt)

        # set selected altitude for descent
        self.swtod = np.where(bs.traf.swvnav * reached_tod * (self.phase >= PHASE['CR']), True, self.swtod)
        bs.traf.selalt = np.where(self.swtod, bs.traf.actwp.nextaltco, bs.traf.selalt)

        # first find which type of descent phase each a/c is in
        idx_climb = np.where(self.descenttype == -1)[0]
        idx_level = np.where(self.descenttype == 0)[0]
        idx_perf = np.where(self.descenttype == 1)[0]
        idx_geo = np.where(self.descenttype == 2)[0]
        idx_rate = np.where(self.descenttype == 3)[0]

        # Determine Vertical Speed
        # _____________________________________________Level flight_____________________________________________________
        self.vs[idx_level] = np.zeros(len(bs.traf.id))[idx_level]

        # __________________________________________Performance Descent_________________________________________________
        # set vs at -999 m/s, this value will be limited by the performance model
        self.vs[idx_perf] = np.where(self.startdescent, -999, np.array([0]))[idx_perf]

        # ___________________________________________Geometric descent__________________________________________________
        # find difference between desired altitude on glideslope and current altitude
        #   used for glideslope tracking with P-controller
        des_alt = np.tan(np.radians(self.descentslope)) * self.dist2wp + bs.traf.actwp.nextaltco\

        d_alt_geo = des_alt - bs.traf.alt
        self.vs[idx_geo] = np.where(self.startdescent,
                                    np.radians(self.descentslope) * bs.traf.gs + d_alt_geo * self.p_gs,
                                    np.array([0]))[idx_geo]

        # __________________________________________Fixed rate descent__________________________________________________
        self.vs[idx_rate] = bs.traf.selvs[idx_rate]

        #_________________________________________________Climb_________________________________________________________
        self.vs[idx_climb] = np.full(len(bs.traf.id),999)[idx_climb]


        # geometric descent is used by performance model to decide on flight strategy
        self.geodescent = np.where(np.isin(self.descenttype, [2,3]), self.steepness, -999)
        # bs.traf.actwp.vs = np.where(self.geodescent, self.steepness * bs.traf.gs, -999)
        self.selvs = np.where(abs(bs.traf.selvs) > 0.1, bs.traf.selvs, self.vsdef)       # given in m/s


        # Run VNAV logic for each aircraft after waypoint shift
        for i in range(len(bs.traf.id)):
            self.vnav.update(self.route, i, self.idxreached)

        self.alt = bs.traf.selalt

        self.vnavvs = np.where(self.swvnavvs, bs.traf.actwp.vs, self.vnavvs)

        # Check if cruise phase has been reached, if so set speed to cruise speed
        bs.traf.selspd = np.where(self.phase == PHASE['CR'], self.cspd, bs.traf.selspd)

        # Below crossover altitude: CAS=const, above crossover altitude: Mach = const
        self.tas = vcasormach2tas(bs.traf.selspd, bs.traf.alt)

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


        # Determine if the speed schedule should be used
        # if own speed is slower than the speed schedule own speed is used
        use_speedschedule = (self.spdschedule < bs.traf.actwp.spdcon) * self.swtod

        # ADDED During descent, speed schedule can be overwritten if own speed is already slower than schedule
        self.spdschedule = np.where(
            (vcasormach2tas(self.spdschedule, bs.traf.alt) > (1.01 * vcasormach2tas(bs.traf.selspd, bs.traf.alt))) *
            (bs.traf.selspd > 1), bs.traf.selspd, self.spdschedule)

        # Where we don't have a turn waypoint, as in turn idx is negative, then put distance
        # as Earth circumference.
        self.dist2turn = np.where(bs.traf.actwp.nextturnidx >= 0, dist2turn, 40075000)

        # Check also whether VNAVSPD is on, if not, SPD SEL has override for next leg
        # and same for turn logic
        usenextspdcon = ((self.dist2wp < dxspdconchg) * (bs.traf.actwp.nextspd > -990.) *
                         bs.traf.swvnavspd * bs.traf.swvnav * bs.traf.swlnav *
                         (self.spdschedule > (bs.traf.actwp.nextspd * self.swtod)))

        useturnspd = np.logical_or(bs.traf.actwp.turntonextwp,
                                   (self.dist2wp < dxturnspdchg + bs.traf.actwp.turndist) *
                                   swturnspd * bs.traf.swvnavspd * bs.traf.swvnav * bs.traf.swlnav)

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
        bs.traf.selspd = np.where(useturnspd, bs.traf.actwp.turnspd,
                                  np.where(usenextspdcon, bs.traf.actwp.nextspd,
                                           np.where((bs.traf.actwp.spdcon >= 0) * bs.traf.swvnavspd,
                                                    np.where(use_speedschedule, self.spdschedule, bs.traf.actwp.spdcon),
                                                    np.where(np.logical_and(bs.traf.swvnavspd, self.swtod),
                                                             self.spdschedule,bs.traf.selspd))))


        # Temporary override when still in old turn
        bs.traf.selspd = np.where(
            inoldturn * (bs.traf.actwp.oldturnspd > 0.) * bs.traf.swvnavspd * bs.traf.swvnav * bs.traf.swlnav,
            bs.traf.actwp.oldturnspd, bs.traf.selspd)

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

        self.EEI_IAS, self.EEI_ROC, self.EEI_MACH, self.EEI_COL = (
            self.EEI.IAS_ROC(self.TOdf, bs.traf.alt, bs.traf.cas, self.sd_spd, self.sd_roc, self.calt))


    @stack.command(name='TO', annotations="acid,txt,txt,alt,spd,float,float")
    def TOcmd(self, idx: 'acid', SID: 'SID' = None, dest: 'dest' = None, calt: 'alt' = 0, cspd: 'spd' = 0,
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
                stack.echo(f"Cruise altitude not provided, FL{int(calt_FL)} used instead")

            if calt_FL > np.nanmin([self.caltdf[idx]['max_calt'].max(), self.TOdf[idx]['ALT'].max()]):
                calt_FL = int(np.nanmin([self.caltdf[idx]['max_calt'].max(), self.TOdf[idx]['ALT'].max()]))
                calt = calt_FL * (ft * 100)
                stack.echo(f"Cruise altitude is too high, FL{int(calt_FL)} used instead")

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
        '''
        Function: Calculate the speeds corresponding to the speed schedule of pyBADA
        For either jet/turboprop or piston aircraft
        Only for descent currently!

        Created by: Winand Mathoera
        Date: 26/09/2022

        Edited by: Albert Chou
        Date: July 2025
        '''


        alt = alt / ft

        # # Correction for minimum landing speed for different performance model
        # if PerfBase.selected().__name__.lower() == "openap":
        #     corr = 1
        # else:
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
        l13 = bs.traf.perf.casdes
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


    @stack.command(name = "ALT")
    def selaltcmd(self, idx: 'acid', alt: 'alt', vspd: 'vspd'=None, own_nav: 'onoff' = False):
        """ ALT acid, alt, [vspd], [own_nav]
            own_nav flag: continue descent along VNAV until the specified altitude
            if no arguments given: start performance descent right now

            Select autopilot altitude command.
            Args:
                    idx:        a/c index
                    alt:        altitude
                    vspd:       optional vertical speed
                    own_nav:    optional boolean; follow own VNAV until this altitude

            Edited by: Teun Vleming
            Date: 03-2024

            Edited by: Arham Elahi
            Date: October 2024
        """

        bs.traf.selalt[idx] = alt

        # Check for optional VS argument
        if vspd:
            bs.traf.swvnav[idx] = False
            if vspd < 0:
                # Use rate descent
                self.vs[idx] = vspd
                self.descenttype[idx] = 3
                self.startdescent[idx] = True
            else:
                self.vs[idx] = vspd
                self.descenttype[idx] = 1
        else:
            if not isinstance(idx, Collection):
                idx = np.array([idx])
            delalt = alt - bs.traf.alt[idx]
            # Check if vs and alt are not contradicting each other, if so set vs to zero
            oppositevs = np.logical_and(bs.traf.selvs[idx] * delalt < 0., abs(bs.traf.selvs[idx]) > 0.01)
            bs.traf.selvs[idx[oppositevs]] = 0.
            # Check if aircraft is descending
            if delalt < 0:
                if own_nav:
                    # descent along VNAV until this altitude
                    bs.traf.swvnav[idx] = True
                    self.startdescent[idx] = False
                    self.cleared_alt = alt
                else:
                    # Descent now with performance descent
                    self.vs[idx] = -999
                    self.descenttype[idx] = 1
                    bs.traf.swvnav[idx] = False
                    self.startdescent[idx] = True
            else:
                if own_nav:
                    # Adjust the cruise altitude, keep using the climb profile from EEI
                    self.calt[idx] = alt
                else:
                    # disregard the profile
                    self.descenttype[idx] = -1
                    bs.traf.swvnav[idx] = False

    @stack.command(name='VS')
    def selvspdcmd(self, idx: 'acid', vspd: 'vspd'):
        """ VS acid,vspd (ft/min)
            Vertical speed command (autopilot) """
        bs.traf.selvs[idx] = vspd  # [fpm]
        bs.traf.swvnav[idx] = False
        self.TOsw[idx] = False

    @stack.command(name='SPD', aliases=("SPEED",))
    def selspdcmd(self, idx: 'acid', casmach: 'spd'):  # SPD command
        """ SPD acid, casmach (= CASkts/Mach)

            Select autopilot speed. """
        # Depending on or position relative to crossover altitude,
        # we will maintain CAS or Mach when altitude changes
        # We will convert values when needed
        bs.traf.selspd[idx] = casmach
        self.setspd[idx] = casmach
        # Used to be: Switch off VNAV: SPD command overrides
        if bs.traf.swvnavspd[idx]:
            self.spddismiss[idx] = True  # to later turn back on swvnavspd
        bs.traf.swvnavspd[idx] = False

        return True

    @stack.command(name='DEST')
    def setdest(self, acidx: 'acid', wpname: 'wpt' = None):
        ''' DEST acid, latlon/airport

            Set destination of aircraft, aircraft wil fly to this airport. '''
        if wpname is None:
            return True, 'DEST ' + bs.traf.id[acidx] + ': ' + self.dest[acidx]
        route = self.route[acidx]
        apidx = bs.navdb.getaptidx(wpname)
        if apidx < 0:
            if bs.traf.ap.route[acidx].nwp > 0:
                reflat = bs.traf.ap.route[acidx].wplat[-1]
                reflon = bs.traf.ap.route[acidx].wplon[-1]
            else:
                reflat = bs.traf.lat[acidx]
                reflon = bs.traf.lon[acidx]

            success, posobj = txt2pos(wpname, reflat, reflon)
            if success:
                lat = posobj.lat
                lon = posobj.lon
            else:
                return False, "DEST: Position " + wpname + " not found."

        else:
            lat = bs.navdb.aptlat[apidx]
            lon = bs.navdb.aptlon[apidx]

        self.dest[acidx] = wpname
        iwp = route.addwpt(acidx, self.dest[acidx], route.dest,
                           lat, lon, 0.0, 0.0)
        # If only waypoint: activate
        if (iwp == 0) or (self.orig[acidx] != "" and route.nwp == 2):
            bs.traf.actwp.lat[acidx] = route.wplat[iwp]
            bs.traf.actwp.lon[acidx] = route.wplon[iwp]
            bs.traf.actwp.nextaltco[acidx] = route.wpalt[iwp]
            bs.traf.actwp.spd[acidx] = route.wpspd[iwp]

            bs.traf.swlnav[acidx] = True
            bs.traf.swvnav[acidx] = True
            route.iactwp = iwp
            route.direct(acidx, route.wpname[iwp])

        # If not found, say so
        elif iwp < 0:
            return False, ('DEST position' + self.dest[acidx] + " not found.")

    @stack.command(name='VNAV')
    def setVNAV(self, idx: 'acid', flag: 'bool' = None):
        """ VNAV acid,[ON/OFF]

            Switch on/off VNAV mode, the vertical FMS mode (autopilot) """
        if not isinstance(idx, Collection):
            if idx is None:
                # All aircraft are targeted
                bs.traf.swvnav = np.array(bs.traf.ntraf * [flag])
                bs.traf.swvnavspd = np.array(bs.traf.ntraf * [flag])
            else:
                # Prepare for the loop
                idx = np.array([idx])

        # Set VNAV for all aircraft in idx array
        output = []
        for i in idx:
            if flag is None:
                msg = bs.traf.id[i] + ": VNAV is " + "ON" if bs.traf.swvnav[i] else "OFF"
                if not bs.traf.swvnavspd[i]:
                    msg += " but VNAVSPD is OFF"
                output.append(bs.traf.id[i] + ": VNAV is " + "ON" if bs.traf.swvnav[i] else "OFF")

            elif flag:
                if not bs.traf.swlnav[i]:
                    return False, (bs.traf.id[i] + ": VNAV ON requires LNAV to be ON")

                route = self.route[i]
                if route.nwp > 0:
                    bs.traf.swvnav[i] = True
                    bs.traf.swvnavspd[i] = True
                    self.route[i].calcfp()
                    actwpidx = self.route[i].iactwp
                    # self.ComputeVNAV(i, self.route[i].wptoalt[actwpidx], self.route[i].wpxtoalt[actwpidx], \
                    #                self.route[i].wptorta[actwpidx], self.route[i].wpxtorta[actwpidx])
                    # bs.traf.actwp.nextaltco[i] = self.route[i].wptoalt[actwpidx]


                else:
                    return False, ("VNAV " + bs.traf.id[i] + ": no waypoints or destination specified")
            else:
                bs.traf.swvnav[i] = False
                bs.traf.swvnavspd[i] = False
        if flag == None:
            return True, '\n'.join(output)