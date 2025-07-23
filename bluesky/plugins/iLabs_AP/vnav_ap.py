from math import floor
import numpy as np
import bluesky as bs
from .descent import Descent
from bluesky.traffic.performance.iLP.performance import PHASE
from bluesky.tools.aero import ft, fpm, vmach2cas, vcasormach2tas, vtas2cas, g0
from bluesky.tools.geo import latlondist, qdrdist


class VNAV:
    """
    VNAV implementation
    in autopilot import:            from bluesky.plugins.iLabs_AP.new_VNAV import VNAV
    in autopilot __init__(self):    self.vnav = VNAV()

    Description:
        Provide the VNAV logic to the autopilot.py
        Provide VNAV logic based on performance phases, for climb phase follows the EEI tables. For descent phase,
        find descent profile between waypoints in a route. Adhere to waypoint constraints. Needs Descent class for
        Top of Descent calculations. Try performance-based descent first, else use geometric descent.
        Returns parameters of route legs, set variables in traffic.autopilot()
    """

    def update(self, route, i, idxreached):
        """"
        Updates the flight plan for the a/c
        For Climb phase uses EEI performance
        Backtracks through wpts to find descent properties
        Descent is divided in legs inbetween restrictive waypoints
        """
        # Add waypoint when it passes th
        r = route[i]
        r.legs = []

        i_ownpos = r.iactwp -1

        # ==============================================================================================================
        # The following lines consider with the implementation of the EEI function required for the realistic
        # take-off and climb performance if TO command is used.
        # ==============================================================================================================
        if (bs.traf.ap.phase[i] < PHASE['CR']) and bs.traf.ap.TOsw[i]:

            # Set VNAV off, so it follows the EEI Climb performance
            bs.traf.swvnav[i] = False

            # Timed function which every dt seconds aims to update the self.EEI_IAS, self.EEI_ROC, self.EEI_MACH and
            # self.EEI_COalt arrays.
            bs.traf.ap.ESTspeeds()

            # Check whether the agent has a speed constraint. If yes, use the minimum of the speed constraint and
            # estimation in the IAS array.
            spdcon = np.logical_not(bs.traf.actwp.nextspd[i] <= 0)
            bs.traf.ap.EEI_IAS[i] = np.where(spdcon, np.minimum(bs.traf.actwp.nextspd[i], bs.traf.ap.EEI_IAS[i]),
                                             bs.traf.ap.EEI_IAS[i])

            # Check whether the agent has a speed constraint. If yes, yse the minimum of the speed constraint and
            # estimation in the IAS array
            spdcon = np.logical_not(bs.traf.ap.setspd[i] <= 0)
            bs.traf.ap.EEI_IAS[i] = np.where(spdcon, np.minimum(bs.traf.ap.setspd[i], bs.traf.ap.EEI_IAS[i]),
                                             bs.traf.ap.EEI_IAS[i])

            # Check for CAS/MACH crossover altitude (COL)
            bs.traf.ap.EEI_SPD[i] = np.where(bs.traf.alt[i] >= (bs.traf.ap.EEI_COalt[i] * 100 * ft),
                                             bs.traf.ap.EEI_MACH[i], bs.traf.ap.EEI_IAS[i])

            # Selected speed is equal to the estimation in the array.
            bs.traf.selspd[i] = np.where(bs.traf.ap.TOsw[i], bs.traf.ap.EEI_SPD[i], bs.traf.selspd[i])

            # # These switches are required to prohibit aircraft from taking off as soon as they have a velocity greater
            # # than 0. Select either 0 or the estimated ROC.
            sw_vs_restr = abs(bs.traf.ap.alt[i] - bs.traf.alt[i]) > 10
            bs.traf.vs[i] = np.where(bs.traf.ap.TOsw[i] * bs.traf.ap.phase[i] == PHASE['TO'],
                                  np.where(sw_vs_restr,
                                           np.where(bs.traf.ap.TOsw[i], bs.traf.ap.EEI_ROC[i], bs.traf.ap.selvs[i]), 0),
                                  bs.traf.vs[i])

            # Update vertical speed
            bs.traf.ap.vs[i] = np.where(bs.traf.ap.TOsw[i], bs.traf.ap.EEI_ROC[i], bs.traf.ap.vs[i])

            # Set cruise altitude after last active waypoint in climb
            bs.traf.selalt[i] = np.where(abs(bs.traf.alt[i] - bs.traf.selalt[i]) < 10,
                                         bs.traf.ap.calt[i], bs.traf.selalt[i])

            # When a/c arrives at ToC, VNAV on and turn TO switch off
            if bs.traf.ap.swtoc[i]:
                bs.traf.swvnav[i] = True
                bs.traf.swvnavspd[i] = True
                bs.traf.ap.TOsw[i] = False
        # ==============================================================================================================


        # ==============================================================================================================
        # The following lines consider with the implementation of performance descent profile
        # The implementation for backtrack through wpts to find descent properties is only called upon when the phase
        # is at cruise or after
        # Descent is divided in legs inbetween restrictive waypoints
        # ==============================================================================================================
        if bs.traf.ap.phase[i] >= PHASE['CR']:
            bs.traf.ap.spdschedule[i] = Descent.descent_speedschedule(i, bs.traf.alt[i])
            for i in idxreached:
                print(r.wpname[i_ownpos])
                # Set current altitude to current position if waypoint altitude
                if r.wpalt[i_ownpos] < -1:
                    r.wpalt[i_ownpos] = bs.traf.alt[i]
                    r.wpaltres[i_ownpos] = "AT"

                j = 1                           # index of wpt in route
                # backtrack from last wpt: create legs between restrictive wpts
                while True:                     # loop through wpts
                    leg_dist = 0
                    end_alt = r.wpalt[-j]

                    # try performance descent
                    type_descent = 1
                    # types:    0: level
                    #           1: perf
                    #           2: geo
                    #           3: rate (unused for now)

                    k = 0
                    while True:                 # backtrack from wpt j to current position wpt
                        k += 1
                        # find distance between 2 wpts
                        leg_dist += latlondist(r.wplat[-j - k + 1], r.wplon[-j - k + 1],
                                               r.wplat[-j - k], r.wplon[-j - k])

                        start_alt = r.wpalt[-j - k]
                        delta_alt = end_alt - start_alt

                        # if wpt is a runway, make a geometric descent with a fixed glideslope:
                        #   -from previous wpt position if it is unrestricted
                        #   -from previous wpt altitude if it is restricted
                        # for wpt types see route.py
                        # caution: EHAM_36R is not considered a runway but EHAM/RWY36R is
                        if r.wptype[-j] == 5 or r.wptype[-j] == 3:
                            type_descent = 2
                            slope = 3.0                         # Standard glideslope to runway [deg]
                            if r.wpaltres[-j - k] == '':
                                descent_dist = float(leg_dist)
                                prev = r.wpalt[np.where(r.wpalt > np.array([0]))[0][-2]]  # previous altitude restriction
                                r.wpalt[-j - k] = np.min((np.tan(np.radians(slope)) * descent_dist, prev))
                                r.wpaltres[-j - k] = 'G'  # make the unrestricted wpt a Geometric wpt
                            else:
                                descent_dist = float(r.wpalt[-j - k] / np.tan(np.radians(slope)))
                            delta_alt = end_alt - r.wpalt[-j - k]
                            leg_data = np.array([r.wpname[-j], r.wpname[-j-k], delta_alt, descent_dist,
                                                 leg_dist, type_descent, slope], dtype = object)
                            r.legs.append(leg_data)
                            j += k
                            break  # end the leg here

                        if start_alt < 0:               # disregard unrestricted wpt
                            if r.wpname[int(r.nwp - j - k)] == r.wpname[i_ownpos]:      # do not extend if wpt index j+k is out of bounds
                                descent_dist = float(leg_dist)
                                slope = 0
                                if delta_alt == 0:
                                    type_descent = 0
                                leg_data = np.array([r.wpname[-j], r.wpname[-j-k], delta_alt, descent_dist,
                                                     leg_dist, type_descent, slope], dtype = object)
                                r.legs.append(leg_data)
                                j += k
                                break
                            else:
                                continue                # extend leg with another wpt

                        # find distance to descent from alt restriction to final wpt alt
                        if delta_alt > 20:
                            climb = True
                            descent_dist = 0
                            type_descent = -1
                        elif delta_alt < -20:
                            descent = True
                            descent_dist = float(self.descentpath(i, delta_alt, j, k, r))
                        else:
                            level = True
                            type_descent = 0
                            descent_dist = 0

                        # check if wpt constraints can be met
                        cont, slope = self.check_wp_constraint(r, j, k, descent_dist,
                                                               leg_dist, delta_alt, i_ownpos)

                        if cont == False:  # constraints can be met but are restricting the leg
                            leg_data = np.array([r.wpname[-j], r.wpname[-j-k], delta_alt, descent_dist,
                                                 leg_dist, type_descent, slope], dtype = object)
                            r.legs.append(leg_data)
                            j += k
                            break
                        elif cont == True:  # constraints can be met, extend leg with another wpt
                            continue
                        elif cont == -1:  # constraints cannot be met; force geometric descent, recheck intermediate wpts
                            # backtrack from wpt -j to wpt -j-k,
                            # find leg distance = tmpdist
                            # descent distance = descdist
                            tmp_leg_dist = 0

                            for l in np.arange(k) + 1:  # loop through wpts
                                tmp_leg_dist += latlondist(r.wplat[-j - l + 1], r.wplon[-j - l + 1],
                                                           r.wplat[-j - l], r.wplon[-j - l])
                                d_alt = end_alt - r.wpalt[-j - l]
                                desc_dist = leg_dist * d_alt / delta_alt
                                if r.wpalt[-j - l] < 0:
                                    continue  # skip unconstrained wpt

                                cont, slope = self.check_wp_constraint(r, j, l, desc_dist,
                                                                       tmp_leg_dist, d_alt, i_ownpos)

                                if cont == True:  # continue backtracking the geo descent
                                    continue
                                else:  # if the intermediate wpt is restrictive, add another GEO descent
                                    k = l
                                    break

                            type_descent = 2
                            descent_dist = desc_dist
                            leg_dist = tmp_leg_dist
                            leg_data = np.array([r.wpname[-j], r.wpname[-j-k], d_alt, descent_dist,
                                                 leg_dist, type_descent, slope], dtype = object)
                            r.legs.append(leg_data)
                            j += k
                            break

                    if r.nwp - j <= r.iactwp - 1:  # quit after current active wpt is reached
                        self.descent_params(r, i, i_ownpos)  # set active leg parameters in traf.ap vectors
                        break
                print(r.legs)
                return




    @staticmethod
    def descent_params(r, i, i_ownpos):
        """"
        Args:
            r:          route object
            i:          a/c/ index
            i_ownpos:   ownpos wpt index

        Returns:
            sets the autopilot descent parameters, adds hysteresis for consistent descent strategy
        """
        bs.traf.ap.descenttype[i] = r.legs[-1][5]           # type of descent (0=level, 1=perf, 2=geo)
        bs.traf.ap.descentslope[i] = r.legs[-1][6]          # slope of geo descent (0 if n/a)
        bs.traf.ap.descentalt[i] = r.legs[-1][2]            # altitude difference of the descent
        bs.traf.ap.descentleg[i] = r.legs[-1][4]

        dist2wp = latlondist(r.wplat[i_ownpos], r.wplon[i_ownpos],
                             r.wplat[i_ownpos + 1], r.wplon[i_ownpos + 1])  # distance from ownpos to active wpt

        if r.legs[-1][4] == r.legs[-1][3]:
            # continue descending when descent length is equal to leg distance
            # value set very large to prevent flying level when wpt is selected early (e.g. when a large turn radius and
            # a large heading change are present at high speeds)
            bs.traf.ap.dist2vs[i] = 999999
        else:
            bs.traf.ap.dist2vs[i] = dist2wp - (r.legs[-1][4] - r.legs[-1][3]) # Distance from active wpt to ToD

        bs.traf.actwp.nextaltco[i] = r.wpalt[r.wpname.index(r.legs[-1][0])]
        return

    @staticmethod
    def check_wp_constraint(r, j, k, descent_dist, leg_dist, delta_alt, i_ownpos):
        """
        Check if the waypoint constraints can be met. If not: force a geometric descent between wpt -j and
        the offending wpt. Then recheck the intermediate wpts.
        """

        tolerance = 2000            # allows for errors in ToD calculation without changing descent strategy

        slope = abs(np.degrees(np.arctan(delta_alt / leg_dist)))
        wp_type = r.wpaltres[-j - k]

        # check if previous wpt constraints can be met
        extend = ((wp_type == "A" and descent_dist <= leg_dist + tolerance) or
                  (wp_type == "B" and descent_dist >= leg_dist - tolerance))

        if extend:
            if r.nwp - j - k == i_ownpos:       # do no extend if wpt index j+k is out of bounds
                cont = False
            else:
                cont = True
        else:
            # AT constraint is met: end the leg here
            # B constraint can be met but is restrictive: end the leg here
            # any wpt constraint is failed: GEO descent
            if ((wp_type == "AT" and descent_dist <= leg_dist + tolerance) or
                    (wp_type == "B" and descent_dist < leg_dist)):
                cont = False
            else:
                cont = -1
        slope = min(slope, 4.5)         # do not allow glide slope greater than 4.5 degrees
        return cont, slope

    def descentpath(self, idx, delta_alt, j, k, r):
        """
        Function: Calculate the descent path of an aircraft to determine the top of descent.
        Ordinary waypoint: waypoint without altitude constraint
        Restricted waypoint: waypoint with an altitude constraint ( AT, AT/BELOW, AT/ABOVE)
        Created by: Winand Mathoera
        Date: 12/10/2022

        Adapted by: Teun Vleming
        Date: 08-02-2024

        Adapted by: Albert Chou
        Date: July 2025

        Args:
            idx:            index of a/c
            delta_alt:      altitude difference
            j:              -j is index of last wpt in backtrack
            k:              number of wpts in backtrack
            r:              route object

        Returns:
            distance it takes to descent a certain delta alt
        """

        self.seg_height = 100           # height of each segment over which the distance is integrated

        # Altitude segments and corresponding glideslope angle (gamma)
        concas = -1

        start_alt = r.wpalt[-j] - delta_alt
        end_alt = r.wpalt[-j]
        wind = bs.traf.wind.getdata(r.wplat[-j - k], r.wplon[-j - k], start_alt)

        bearing = r.wpdirfrom[-j]

        # call descent class
        ds = Descent(idx, start_alt, end_alt, concas, self.seg_height)

        # segment index of current altitutde
        alti = -1

        vsaxalt, vsaxdist = self.verticalax(ds.rod[alti], ds.tasspeeds[alti])

        # Predicted distance needed to reach altitude of restricted waypoint
        descent_distance = self.descentdistance(idx, wind, bearing, ds) + vsaxdist
        return descent_distance

    @staticmethod
    def gammatas_wind(wind, bearing, rod, tas):
        '''
        Function: Update gammastas for new wind vector and aircraft heading
        Args:
            wind:       wind vector
            bearing:    waypoint bearing
            rod:        rate of descent
            tas:        true airspeed

        Returns:
            windbearing and glideslope angle with consideration of wind
        '''
        windbearing = (wind[0] * np.sin(np.radians(bearing))) + (wind[1] * np.cos(np.radians(bearing)))

        # recalculate the gammatas angle using the horizontal and vertical components
        gammatas = np.arcsin(rod / (tas + windbearing))
        return windbearing, gammatas

    def descentdistance(self, idx, wind, bearing, ds):
        """
        Function: Calculate distance needed to travel between two different altitudes using
        nominal descent rates (gammaTAS)
        Args:
            idx:        a/c index
            wind:       wind vector at start of descent
            bearing:    track bearing
            ds:         descent class

        Edited by: Arham Elahi
        Date: September 2024
        """
        wind_bearing, gammatas = self.gammatas_wind(wind, bearing, ds.rod, ds.tasspeeds)

        dist = sum(self.seg_height * ft / np.tan(-gammatas))

        # Add the deceleration segment required for transition from cruise speed to descent speed schedule
        if ~np.isnan(bs.traf.ap.calt[idx]) and ~np.isnan(bs.traf.ap.cspd[idx]):
            dist += self.decelsegment(ds, idx, bs.traf.ap.calt[idx],
                                      ds.descent_speedschedule(idx, bs.traf.ap.calt[idx]), bs.traf.ap.cspd[idx])

        # loop through deceleration segments
        for i in range(len(ds.decel_altspd)):
            alt, v0, v1 = ds.decel_altspd[i]
            added_dist = self.decelsegment(ds, idx, alt, v0, v1)
            dist += added_dist

        return dist

    @staticmethod
    def verticalax(rod, spd, ax = 300 * fpm):
        '''
        Function: Account for vertical acceleration starting from 0 vertical speed
        '''

        alt = (0 + rod) / 2 * rod / ax  # added altitude based on average of 0 and desired ROD
        dist = spd * abs(rod) / ax  # added distance during acceleration based on current a/c speed
        return alt, dist

    def decelsegment(self, ds, idx, alt, v0 ,v1):
        '''
        Function: Determine added distance and altitude from deceleration segment. Simulate the deceleration,
        compare distance to nominal (instantaneous deceleration) distance and sum the extra distances for each step.
        Args:
            ds:         descent class
            idx:        a/c id
            alt:        altitude of a/c
            v0:         calibrated airspeed to be reached [m/s]
            v1:         current calibrated airspeed [m/s]
            gammatas:   array of gammatas angles
            wind:       speed of wind in flight direction m/s
        '''

        # multiplier to speed up calculations (set to 1 as base step)
        m = 1

        add_alt, new_dist, old_dist = 0, 0, 0

        # convert speed to cas if Mach
        if v1 < 3:
            v1 = vmach2cas(v1, alt)

        if v0 < 3:
            v0 = vmach2cas(v0, alt)

        acc = v1 < v0             # Check if acceleartion is needed (True if accelerating)

        # While CAS or Mach not reached yet
        while True:

            v0tas = vcasormach2tas(v0, alt + add_alt)
            v1tas = vcasormach2tas(v1, alt + add_alt)

            # find difference
            delspd = v0tas - v1tas

            # determine if acceleration is needed (transition longer than sim-timestep)
            need_ax = np.abs(delspd) > np.abs(bs.sim.simdt * m * bs.traf.perf.axmax[idx])

            # determine acceleartion:
            ax = need_ax * np.sign(delspd) * bs.traf.perf.axmax[idx]

            # calculate new TAS and CAS
            tas = np.where(need_ax, v1tas + ax * bs.sim.simdt * m , v0tas)
            cas = vtas2cas(tas, alt + add_alt)

            # find new ROD
            T, D, data = ds.TandD(idx, alt + add_alt, tas, ds.phases(alt + add_alt, alt ,bs.traf.ap.calt[idx]))
            rod = ((T - D ) * tas) / (bs.traf.perf.mass[idx] * g0) * 0.3

            # flight path angles
            gamma_new = np.arcsin(rod / v1tas)

            # calculate added altitude and added distance
            add_alt += rod * bs.sim.simdt

            new_dist += rod * bs.sim.simdt / np.tan(gamma_new)

            # update speed for loop
            v1 = cas

            if acc:
                if np.round(v1, 5) >= np.round(v0, 5):
                    break
            else:
                if np.round(v1, 5) <= np.round(v0, 5):
                    break

        # find index of deceleration segment in ds altitude array and add up distance traveled for each segment
        for i in range(int(np.floor(-add_alt/ft/self.seg_height)), -1, -1):
            n = np.argmin(np.absolute(alt - i * self.seg_height * ft - ds.segments))
            gamma_old = ds.gamma[n]
            if i != 0:
                old_dist += self.seg_height * ft / np.tan(-gamma_old)
                add_alt += self.seg_height * ft
            else:
                old_dist += -add_alt / np.tan(gamma_old)
        add_dist = new_dist - old_dist
        return add_dist