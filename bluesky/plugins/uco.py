""" UCO: which unit controls each flight and which units it reaches next. Units nobody works in
    an ATR are simulated. Needs scenario/iLabs/uco-airspace.scn (PCALL it after PLUGIN UCO). """
import time
from itertools import groupby
import numpy as np
import bluesky as bs
from bluesky import core, stack, traf
from bluesky.network.publisher import state_publisher
from bluesky.tools import areafilter, geo
from bluesky.tools.aero import ft, nm, fpm

UNITS = ('RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'IN2', 'IN3', 'RAA', 'RAR', 'RAS', 'APP', 'FIC', 'MIL', 'MUAC', 'EXT')
ACC = ('RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'IN2', 'IN3')
TOC_NAMES = {'ATP': 'RAA', 'RIV': 'RAR', 'SUG': 'RAS'}
STACKS = ('RAA', 'RAR', 'RAS')
PRIORITY = ('APP', 'IN2', 'IN3', 'RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'MUAC') # where areas overlap, the first wins
INBOUND_ONLY = ('IN2', 'IN3')

PREDICT_EVERY = 5. # [s]
LOOKAHEAD = 25 * 60 # [s]
HEADING_LOOKAHEAD = 10 * 60 # [s] for a flight not on its route
STEP = 30 # [s] between path points
MIN_POINTS = 2 # in a row, for a unit to count
TOC_BEFORE = 120 # [s] a simulated unit hands over this long before the next unit's airspace
DEFAULT_RATE = 2000 * fpm # climb or descent rate of a flight that hasn't started yet
STEEPNESS = 3000 * ft / (10 * nm) # BlueSky's descent slope
ATR_TIMEOUT = 15 # [s] wall time without ATCPOS before an ATR counts as gone
STACK_LEVEL = 100 # [FL] simulated ACC sectors clear Schiphol inbounds down to this
STACK_FIXES = ('ARTIP', 'RIVER', 'SUGOL')
APP_LEVEL = 70 # [FL] an inbound is expected to be at after its stack, inside APP's airspace
ACC_TOP = 240 # [FL] simulated ACC sectors clear climbing flights up to this at most
LANDING_RANGE = 60 # [nm] from a destination other than Schiphol, a simulated unit lets the flight descend to land

uco = None


def init_plugin():
    global uco
    uco = Uco()
    config = {
        'plugin_name': 'UCO',
        'plugin_type': 'sim',
    }
    return config


@state_publisher(topic='UCODATA', dt=1000)
def ucodata():
    if not uco.active():
        return None
    return dict(id=traf.id, uco=uco.uco, assigned=uco.assigned, prev=uco.prev, passed=uco.passed, ahead=uco.ahead)


class Uco(core.Entity):
    def __init__(self):
        super().__init__()
        self.delay = 30.
        self.positions = {} # ATR client id -> (its units, wall time last heard)
        self.last_predict = -1e9
        with self.settrafarrays():
            self.uco = []
            self.assigned = []
            self.prev = []
            self.passed = []
            self.ahead = []
            self.t_next = np.array([]) # [s] until the path is in the next unit's airspace
            self.t_accept = np.array([])
            self.new = np.array([], dtype=bool)
            self.returned = [] # unit that gave it back with PUC
            self.cleared_by = [] # simulated unit that gave the flight its standard clearance
            self.landing = np.array([], dtype=bool)

    def create(self, n=1):
        super().create(n)
        self.uco[-n:] = [''] * n
        self.assigned[-n:] = [''] * n
        self.prev[-n:] = [''] * n
        self.passed[-n:] = [[] for _ in range(n)]
        self.ahead[-n:] = [[] for _ in range(n)]
        self.t_next[-n:] = np.inf
        self.t_accept[-n:] = np.nan
        self.new[-n:] = True
        self.returned[-n:] = [''] * n
        self.cleared_by[-n:] = [''] * n
        self.landing[-n:] = False

    def reset(self):
        super().reset()
        self.delay = 30.
        self.last_predict = -1e9

    @staticmethod
    def active():
        return any(name.startswith('UCO_') for name in areafilter.basic_shapes)

    def worked(self):
        ''' The units a person works in one of the ATRs heard from lately '''
        now = time.time()
        return {unit for units, heard in self.positions.values() if now - heard < ATR_TIMEOUT for unit in units}

    @core.timed_function(name='uco', dt=1)
    def update(self):
        if not self.active() or not traf.ntraf:
            return
        if bs.sim.simt - self.last_predict >= PREDICT_EVERY or self.new.any():
            self.last_predict = bs.sim.simt
            self.predict()

        worked = self.worked()
        for i in range(traf.ntraf):
            following = self.next(i)
            if self.uco[i] not in worked and self.assigned[i] in worked and self.assigned[i] not in self.ahead[i]:
                self.assigned[i] = '' # flight won't reach that person's unit any more
            # simulated unit hands over, but not to a unit that gave it back until it's in its airspace
            if self.uco[i] not in worked and not self.assigned[i] and self.t_next[i] <= TOC_BEFORE \
                    and not (following == self.returned[i] and self.t_next[i] > 0):
                self.assigned[i] = following
            if self.assigned[i] and self.assigned[i] not in worked:
                if np.isnan(self.t_accept[i]):
                    self.t_accept[i] = bs.sim.simt + self.delay
                elif bs.sim.simt >= self.t_accept[i]:
                    self.take(i, self.assigned[i])
            else:
                self.t_accept[i] = np.nan
            if self.uco[i] not in worked:
                self.clear(i)

    def clear(self, i):
        ''' What a simulated unit tells a flight it has: once per unit, and the descent to a nearby airport '''
        acid, unit, dest = traf.id[i], self.uco[i], traf.ap.dest[i]
        if not self.landing[i] and dest not in ('', 'EHAM'):
            apt = bs.navdb.getaptidx(dest)
            if apt >= 0 and geo.kwikdist(traf.lat[i], traf.lon[i], bs.navdb.aptlat[apt], bs.navdb.aptlon[apt]) < LANDING_RANGE:
                self.landing[i] = True
                route = traf.ap.route[i]
                route.wpalt = [-999.] * len(route.wpalt) # the cruise level, else VNAV would climb back to it
                stack.stack(f'VNAV {acid} ON')
        if self.cleared_by[i] == unit or self.landing[i]:
            return
        self.cleared_by[i] = unit
        if dest == 'EHAM':
            if unit == 'APP':
                stack.stack(f'VNAV {acid} ON') # down to land
            elif unit in ACC and traf.selalt[i] > STACK_LEVEL * 100 * ft:
                stack.stack(f'ALT {acid} FL{STACK_LEVEL}')
            return
        # cruise level: the highest level in the route
        route = traf.ap.route[i]
        rfl = round(max((a for a in route.wpalt if a > 0), default=0) / ft / 100)
        # a flight an adjacent centre has from the start keeps the level the scenario gave it
        top = min(rfl, ACC_TOP) if unit in ACC else rfl if unit in ('MUAC', 'EXT') and self.prev[i] else 0
        if top and traf.selalt[i] < top * 100 * ft - 50 * ft:
            stack.stack(f'ALT {acid} FL{top}')

    def next(self, i):
        return next((unit for unit in self.ahead[i] if unit not in STACKS), '')

    def take(self, i, unit):
        if self.uco[i]:
            self.passed[i].append(self.uco[i])
        self.prev[i] = self.uco[i]
        self.uco[i] = unit
        self.assigned[i] = ''
        self.t_accept[i] = np.nan
        self.ahead[i] = [u for u in self.ahead[i] if u != unit]
        self.returned[i] = ''
        self.t_next[i] = np.inf # until the next prediction

    def predict(self):
        ''' Every flight's path for the coming minutes, and the units it passes '''
        paths = [self.path(i) for i in range(traf.ntraf)]
        lat, lon, alt = (np.concatenate([p[k] for p in paths]) for k in range(3))
        inbound = np.concatenate([np.full(len(p[0]), traf.ap.dest[i] == 'EHAM') for i, p in enumerate(paths)])

        units = np.full(len(lat), '', dtype=object)
        for unit in PRIORITY:
            for name, area in areafilter.basic_shapes.items():
                if name == f'UCO_{unit}' or name.startswith(f'UCO_{unit}_'):
                    inside = (units == '') & area.checkInside(lat, lon, alt)
                    if unit in INBOUND_ONLY:
                        inside &= inbound
                    units[inside] = unit

        start = 0
        for i, p in enumerate(paths):
            at = list(units[start:start + len(p[0])])
            start += len(p[0])
            if self.new[i]:
                self.uco[i] = at[0] or 'EXT' # outside all areas: with an adjacent centre
                self.new[i] = False
            # skip the bit still in airspace of units it already had ('' is outside all areas, so EXT)
            gone = {'' if u == 'EXT' else u for u in self.passed[i]}
            first = next((k for k, u in enumerate(at) if u not in gone), len(at))
            rest = at[first:]
            longest = {} # longest run of points per unit
            for u, run in groupby(rest):
                longest[u] = max(longest.get(u, 0), len(list(run)))
            counted = [u for u in rest if u and longest[u] >= MIN_POINTS]
            inside = [first + k for k, u in enumerate(rest) if u]
            if not inside or inside[-1] < len(at) - 1:
                counted.append('EXT')
            self.ahead[i] = [u for u in dict.fromkeys(counted) if u != self.uco[i]]
            if self.uco[i] == 'APP' and traf.ap.dest[i] == 'EHAM':
                self.ahead[i] = [] # lands

            following = self.next(i)
            if following == 'EXT':
                self.t_next[i] = STEP * (inside[-1] + 1 if inside else first)
            elif following:
                self.t_next[i] = STEP * at.index(following, first)
            else:
                self.t_next[i] = np.inf

    def path(self, i):
        ''' lat, lon, alt of flight i every STEP seconds over the look ahead '''
        route = traf.ap.route[i]
        on_route = traf.swlnav[i] and 0 <= route.iactwp < route.nwp
        t = np.arange(0, (LOOKAHEAD if on_route else HEADING_LOOKAHEAD) + 1, STEP)
        dist = traf.gs[i] * t / nm # [nm] flown at each point

        if on_route:
            wplat = np.append(traf.lat[i], route.wplat[route.iactwp:])
            wplon = np.append(traf.lon[i], route.wplon[route.iactwp:])
            along = np.append(0., np.cumsum(geo.kwikdist(wplat[:-1], wplon[:-1], wplat[1:], wplon[1:])))
            lat, lon = np.interp(dist, along, wplat), np.interp(dist, along, wplon)
        else:
            lat, lon = geo.qdrpos(traf.lat[i], traf.lon[i], traf.trk[i], dist)

        # climbs to the highest waypoint level ahead: VNAV does that by itself, without it a controller will.
        # Else the selected level is held
        vnav = on_route and traf.swvnav[i]
        target = traf.selalt[i]
        wpalt = np.array(route.wpalt[route.iactwp:], dtype=float) if on_route else np.zeros(0)
        if len(wpalt):
            target = max(target, wpalt.max())
        inbound = on_route and traf.ap.dest[i] == 'EHAM'
        if inbound:
            target = min(target, STACK_LEVEL * 100 * ft) # will be taken down to the stack
        rate = abs(traf.vs[i]) if abs(traf.vs[i]) > 100 * fpm else DEFAULT_RATE
        alt = traf.alt[i] + np.clip(target - traf.alt[i], -rate * t, rate * t)

        # and goes down into APP after its stack, also once it has passed it
        if inbound:
            k = next((k for k, name in enumerate(route.wpname[route.iactwp:]) if name in STACK_FIXES), None)
            if self.uco[i] == 'APP' or (k is None and any(name in STACK_FIXES for name in route.wpname[:route.iactwp])):
                alt = np.minimum(alt, APP_LEVEL * 100 * ft)
            elif k is not None:
                alt = np.where(dist > along[k + 1], np.minimum(alt, APP_LEVEL * 100 * ft), alt)

        # stay under the descent slope to lower waypoint levels and the destination.
        # a level only counts once past the waypoint before it
        if vnav:
            if route.wpname[-1] == traf.ap.dest[i]:
                wpalt[-1] = 0.
            previous = 0.
            for d, a in zip(along[1:], wpalt):
                if a >= 0:
                    alt = np.where(dist >= previous, np.minimum(alt, a + STEEPNESS * np.maximum(d - dist, 0.) * nm), alt)
                    previous = d
            # but a flight that is too high can't come down faster than BlueSky descends
            alt = np.maximum(alt, traf.alt[i] - max(abs(traf.vs[i]), STEEPNESS * traf.gs[i]) * t)
        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float), np.maximum(alt, 0.)

    def error(self, text):
        ''' Error to the ATR that sent the command, or the normal one for a scenario '''
        if stack.sender():
            bs.net.send('UCOERROR', text)
            return True
        return False, text

    @stack.command
    def ucodelay(self, seconds: float):
        ''' Seconds a simulated unit takes to accept a flight handed to it '''
        self.delay = seconds

    @stack.command
    def atcpos(self, *units: 'txt'):
        ''' Heartbeat of an ATR: the units its pages work '''
        self.positions[stack.sender()] = (set(units), time.time())

    @stack.command(name='UCO')
    def ucocmd(self, acid: 'acid', unit: 'txt'):
        ''' Take control of a flight '''
        if unit not in UNITS:
            return self.error('ATC function is invalid')
        if self.uco[acid] == unit:
            return True
        if self.assigned[acid] and self.assigned[acid] != unit:
            return self.error('Flight assigned to other unit')
        self.take(acid, unit)
        return True

    @stack.command
    def tuc(self, acid: 'acid', unit: 'txt'):
        ''' Take control of a flight, also when it is assigned to another unit '''
        if unit not in UNITS:
            return self.error('ATC function is invalid')
        if self.uco[acid] != unit:
            self.take(acid, unit)
        return True

    @stack.command
    def toc(self, acid: 'acid', unit: 'txt' = ''):
        ''' Hand a flight to a unit, by default the next one '''
        unit = TOC_NAMES.get(unit, unit) or self.next(acid)
        if not unit:
            return self.error('No next unit')
        if unit not in UNITS:
            return self.error('ATC function is invalid')
        if unit == self.uco[acid]:
            return self.error(f'Flight already with {unit}')
        self.assigned[acid] = unit
        self.t_accept[acid] = np.nan
        return True

    @stack.command
    def toccancel(self, acid: 'acid'):
        ''' Cancel a flight's handover: it stays with the unit that has it '''
        self.assigned[acid] = ''
        self.t_accept[acid] = np.nan
        return True

    @stack.command
    def puc(self, acid: 'acid', unit: 'txt'):
        ''' Give a flight back to the unit that had it before '''
        if self.uco[acid] != unit:
            return self.error('Not UCO')
        if self.assigned[acid]:
            return self.error('Flight assigned to other ATCF')
        if not self.prev[acid]:
            return self.error('No previous UCO')
        self.returned[acid] = unit
        self.uco[acid] = self.passed[acid].pop()
        self.prev[acid] = self.passed[acid][-1] if self.passed[acid] else ''
        return True
