""" Datalink: CPDLC dialogues between the controllers and the pilots."""
import numpy as np
import bluesky as bs
from bluesky import core, stack, traf
from bluesky.network.publisher import state_publisher, StatePublisher
from bluesky.tools import geo
from bluesky.tools.aero import ft, nm, fpm
from bluesky.plugins import uco
from openap import WRAP

TIMEOUT = 120 # [s]
REPLY_DELAY = (5, 20) # [s] of the simulated pilot
EXEC_DELAY = (5, 15)
MARGIN = 1.1 # on the rate a crossing needs
PSL_GRACE = 15 # [s] before a wrong selected level alerts
OPEN = ('OPEN', 'STANDBY')
AMBER = ('UNABLE', 'TIMEOUT')
REPLIES = ('WILCO', 'UNABLE', 'STANDBY')
CROSS_WORDS = {'AT': 'AT', 'ABV': 'AT OR ABOVE', 'BLW': 'AT OR BELOW'}
VOICE_TYPES = {'EFL': 'Level', 'DCT': 'Route', 'HDG': 'Route', 'PHD': 'Route', 'CTO': 'Route', 'SPD': 'Speed',
               'TOC': 'Transfer'} # dialogue type an ATR command replaces

FREQUENCIES = {
    'RA1': ('AMSTERDAM ACC', '134.375'),
    'RA2': ('AMSTERDAM ACC', '124.880'),
    'RA3': ('AMSTERDAM ACC', '123.850'),
    'RA4': ('AMSTERDAM ACC', '125.750'),
    'RA5': ('AMSTERDAM ACC', '123.705'),
    'IN2': ('AMSTERDAM ACC', '128.580'),
    'IN3': ('AMSTERDAM ACC', '130.955'),
    'APP': ('SCHIPHOL APPROACH', '119.055'),
    'FIC': ('AMSTERDAM INFORMATION', '124.300'),
    'MIL': ('DUTCH MIL', '125.930'),
    'MUAC': ('MAASTRICHT', '135.960'),
    'EXT': ('EXT', '129.575'),
}

datalink = None
descent_limits = {}


def init_plugin():
    global datalink
    datalink = Datalink()
    config = {
        'plugin_name': 'DATALINK',
        'plugin_type': 'sim',
    }
    return config


@state_publisher(topic='DLDATA', dt=1000)
def dldata():
    if not uco.uco or not uco.uco.active():
        return None
    return dict(id=traf.id, capable=datalink.capable.tolist(), logged_on=datalink.logged_on.tolist(),
                cleared=datalink.cleared.tolist(), spd=datalink.spd.tolist(), dialogues=datalink.dialogues,
                psl=(bs.sim.simt - datalink.psl_since >= PSL_GRACE).tolist())


def descent_limit(actype):
    ''' Fastest descent of a type in WRAP [m/s], None if WRAP doesn't know it '''
    if actype not in descent_limits:
        try:
            descent_limits[actype] = -WRAP(actype).descent_vs_concas()['minimum']
        except ValueError:
            descent_limits[actype] = None
    return descent_limits[actype]


def active(d):
    ''' Waiting for the pilot, waiting to be flown, or amber '''
    return d['state'] in OPEN + AMBER or (d['state'] == 'WILCO' and d['options'])


class Datalink(core.Entity):
    def __init__(self):
        super().__init__()
        self.dialogues = []
        self.viewed = set() # flights open in a pilot panel
        with self.settrafarrays():
            self.capable = np.array([], dtype=bool)
            self.logged_on = np.array([], dtype=bool)
            self.cleared = np.array([], dtype=int) # [FL] accepted by WILCO, 0 when none
            self.cleared_ok = [] # [ft] selected altitudes that keep the cleared level
            self.spd = np.array([], dtype=int) # [kt] given by voice or a flown uplink, 0 when none
            self.cross_level = np.array([], dtype=int) # [FL] first level of a loaded crossing
            self.psl_since = np.array([]) # [s] since the selected level differs from the cleared one
            self.pending = [] # loaded crossing: (point, level after it or 0)

    def create(self, n=1):
        super().create(n)
        self.capable[-n:] = True
        self.logged_on[-n:] = True
        self.cleared[-n:] = 0
        self.cleared_ok[-n:] = [[] for _ in range(n)]
        self.spd[-n:] = 0
        self.cross_level[-n:] = 0
        self.psl_since[-n:] = np.nan
        self.pending[-n:] = [None for _ in range(n)]

    def reset(self):
        super().reset()
        self.dialogues = []
        self.viewed = set()

    @core.timed_function(name='datalink', dt=1)
    def update(self):
        simt = bs.sim.simt
        for d in self.dialogues:
            i = traf.id2idx(d['acid'])
            if i < 0:
                if active(d):
                    self.close(d)
                continue
            if d['state'] == 'OPEN' and simt >= d['t_timeout']:
                d.update(state='TIMEOUT', t_last=simt, t_timeout=None, t_auto=None, auto_left=None)
            elif d['t_auto'] is not None and simt >= d['t_auto']:
                if d['state'] in OPEN:
                    self.reply(d, 'UNABLE' if d['achievable'] is False else 'WILCO')
                elif d['state'] == 'WILCO':
                    try:
                        self.execute(i, d, 'LOAD' if 'LOAD' in d['options'] else 'SET', d['value'])
                    except ValueError:
                        # point already passed, a crossing can still do SET
                        if 'SET' in d['options']:
                            self.execute(i, d, 'SET', d['value'])
                        else:
                            d.update(state='UNABLE', reply='UNABLE', t_last=simt, t_auto=None, auto_left=None)
                elif d.get('air') in OPEN:
                    self.air_reply(d, 'UNABLE')

        # an ALT from somewhere else drops the cleared level (with VNAV on it's just the route levels)
        for i in np.flatnonzero(self.cleared):
            if not traf.swvnav[i] and round(traf.selalt[i] / ft) not in self.cleared_ok[i]:
                self.cleared[i] = 0

        # crossing point passed: normal rate again, and on to the cleared level after an AT
        for i, pend in enumerate(self.pending):
            if pend and self.distance(i, pend[0]) is None:
                stack.stack(f'VS {traf.id[i]} 0')
                if pend[1]:
                    stack.stack(f'ALT {traf.id[i]} FL{pend[1]}')
                self.pending[i] = None

        selected = np.round(traf.selalt / ft / 100)
        differs = (self.cleared > 0) & (selected != self.cleared) & (selected != self.cross_level)
        self.psl_since[~differs] = np.nan
        self.psl_since[differs & np.isnan(self.psl_since)] = simt

    def error(self, text):
        ''' An error for the ATR that sent the command, or the normal BlueSky error for a scenario '''
        if stack.sender():
            bs.net.send('DLERROR', text)
            return True
        return False, text

    def publish(self):
        StatePublisher('DLDATA').send_replace()
        return True

    def find(self, dlg):
        return next((d for d in self.dialogues if d['id'] == dlg), None)

    def schedule(self, d, delays):
        ''' Simulated pilot's next step, on hold while the flight is in a pilot panel '''
        wait = np.random.uniform(*delays)
        if d['acid'] in self.viewed:
            d.update(t_auto=None, auto_left=wait)
        else:
            d.update(t_auto=bs.sim.simt + wait, auto_left=None)

    def close(self, d, state='CLOSED'):
        # was: the state it closed from
        d.update(was=d['state'], state=state, t_last=bs.sim.simt, t_timeout=None, t_auto=None, auto_left=None)

    def finish(self, acid, dtype):
        ''' New instruction: replaces open dialogues of its type, closes amber ones '''
        for d in self.dialogues:
            if d['acid'] == acid and d['dtype'] == dtype and active(d):
                self.close(d, 'CLOSED' if d['state'] in AMBER else 'REPLACED')

    def distance(self, i, point):
        ''' Distance [nm] along the route to a point ahead, or None when it isn't ahead '''
        route = traf.ap.route[i]
        ahead = route.wpname[route.iactwp:] if route.iactwp >= 0 else []
        if point not in ahead:
            return None
        k = route.iactwp + ahead.index(point) + 1
        lat = np.append(traf.lat[i], route.wplat[route.iactwp:k])
        lon = np.append(traf.lon[i], route.wplon[route.iactwp:k])
        return geo.kwikdist(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()

    def rates(self, i, d):
        ''' Rate a crossing needs and the type's limit [m/s], None if n/a. Also kept on the dialogue in fpm '''
        p = d['params']
        target = p['cross'] * 100 * ft
        change = {'AT': abs(target - traf.alt[i]), 'ABV': target - traf.alt[i], 'BLW': traf.alt[i] - target}[p['cond']]
        needed = MARGIN * change * traf.gs[i] / (self.distance(i, p['point']) * nm) if change > 0 else None
        limit = traf.perf.vsmax[i] if target > traf.alt[i] else descent_limit(traf.type[i])
        d.update(achievable=bool(needed is None or limit is None or needed <= limit),
                 rate_needed=None if needed is None else int(needed / fpm),
                 rate_max=None if limit is None else int(limit / fpm))
        return needed, limit

    def message(self, i, cmd, args, unit):
        ''' The dialogue for a CPDLC message, or ValueError with the reason it can't be given '''
        d = dict(acid=traf.id[i], params={}, value='', field='', options=[],
                 achievable=None, rate_needed=None, rate_max=None)
        if cmd == 'EFL':
            level = int(args[0])
            if level == round(traf.alt[i] / ft / 100):
                raise ValueError('Level is the actual level')
            kind = 'CLIMB' if level * 100 * ft > traf.alt[i] else 'DESCEND'
            d.update(kind=kind, dtype='Level', text=f'{kind} TO FL{level:03d}', summary=f'{kind} {level:03d}',
                     params=dict(level=level), value=f'{level:03d}', field='cfl', options=['SET'])
            if len(args) > 1:
                point, cond, cross = args[1], args[2], int(args[3])
                if cond not in CROSS_WORDS:
                    raise ValueError('Invalid crossing')
                if (cond, kind) in (('ABV', 'DESCEND'), ('BLW', 'CLIMB')):
                    raise ValueError(f'{cond} needs a {"climb" if cond == "ABV" else "descent"}')
                now = round(traf.alt[i] / ft / 100)
                if (kind == 'CLIMB' and not now < cross <= level) or (kind == 'DESCEND' and not level <= cross < now):
                    raise ValueError('Crossing level not between actual and cleared level')
                if self.distance(i, point) is None:
                    raise ValueError('Point not in planned route')
                d['text'] += f' CROSS {point} {CROSS_WORDS[cond]} FL{cross:03d}'
                d['summary'] += f' {point} {cond} {cross:03d}'
                d['params'].update(point=point, cond=cond, cross=cross)
                d['options'] = ['LOAD', 'SET']
                self.rates(i, d)
        elif cmd == 'DCT':
            if self.distance(i, args[0]) is None:
                raise ValueError('Point not in planned route')
            d.update(kind='DIRECT', dtype='Route', text=f'PROCEED DIRECT TO {args[0]}', summary=f'DCT {args[0]}',
                     params=dict(point=args[0]), value=args[0], field='dct', options=['LOAD'])
        elif cmd == 'HDG':
            hdg = int(args[0]) % 360 or 360
            d.update(kind='HEADING', dtype='Route', text=f'FLY HEADING {hdg:03d}', summary=f'HDG {hdg:03d}',
                     params=dict(hdg=hdg), value=f'{hdg:03d}', field='hdg', options=['SET'])
        elif cmd == 'PHD':
            d.update(kind='PRESENT', dtype='Route', text='CONTINUE PRESENT HEADING', summary='PHD',
                     field='hdg', options=['SET'])
        elif cmd == 'SPD':
            spd = int(args[0])
            d.update(kind='SPEED', dtype='Speed', text=f'MAINTAIN {spd}', summary=f'SPD {spd}',
                     params=dict(spd=spd), value=str(spd), field='spd', options=['SET'])
        elif cmd == 'TOC':
            to = uco.TOC_NAMES.get(args[0], args[0]) if args else uco.uco.next(i)
            if not to:
                raise ValueError('No next unit')
            if to == uco.uco.uco[i]:
                raise ValueError(f'Flight already with {to}')
            if to not in FREQUENCIES:
                raise ValueError(f'No frequency for {to}')
            name, freq = FREQUENCIES[to]
            d.update(kind='CONTACT', dtype='Transfer', text=f'CONTACT {name} {freq}', summary=f'CONTACT {to}',
                     params=dict(unit=to), value=to, field='callsign')
        elif cmd == 'CHK':
            if unit not in FREQUENCIES:
                raise ValueError(f'No frequency for {unit}')
            d.update(kind='CHECK', dtype='Instruction', text=f'CHECK STUCK MICROPHONE {FREQUENCIES[unit][1]}',
                     summary='CHECK MIC')
        elif cmd == 'CTO':
            to, vias = args[0], list(args[1:])
            if self.distance(i, to) is None:
                raise ValueError('Point not in planned route')
            if any(bs.navdb.getwpidx(via) < 0 for via in vias):
                raise ValueError('Unknown point')
            d.update(kind='CLEARED', dtype='Route', summary=f'CTO {to}',
                     text=f'ROUTE CHANGE --- CLEARED TO {to}' + (' VIA ' + ' '.join(vias) if vias else ''),
                     params=dict(to=to, vias=vias), value=to, field='dct', options=['LOAD'])
        else:
            raise ValueError('Invalid message')
        return d

    def run(self, i, d, value=''):
        ''' Fly a clearance, from the pilot or by voice '''
        acid, p, kind = traf.id[i], d['params'], d['kind']
        if kind in ('CLIMB', 'DESCEND'):
            level = int(value) if value else p['level']
            crossing = 'point' in p and not value # SET flies just the level
            if crossing and self.distance(i, p['point']) is None:
                raise ValueError(f'{p["point"]} already passed')
            cmds = [f'VS {acid} 0'] if self.pending[i] else []
            self.pending[i] = None
            self.cleared_ok[i].append(level * 100)
            if crossing:
                needed, limit = self.rates(i, d)
                rate = needed if limit is None or needed is None else min(needed, limit)
                first = p['cross'] if p['cond'] == 'AT' else level # AT levels off at the crossing level first
                cmds.append(f'ALT {acid} FL{first}' + (f' {rate / fpm:.0f}' if rate and rate > traf.ap.vsdef[i] else ''))
                self.pending[i] = (p['point'], level if p['cond'] == 'AT' else 0)
                self.cleared_ok[i].append(first * 100)
                self.cross_level[i] = first
            else:
                cmds.append(f'ALT {acid} FL{level}')
        elif kind == 'DIRECT':
            if self.distance(i, p['point']) is None:
                raise ValueError(f'{p["point"]} already passed')
            cmds = [f'DIRECT {acid} {p["point"]}']
        elif kind == 'HEADING':
            cmds = [f'HDG {acid} {int(value) if value else p["hdg"]}']
        elif kind == 'PRESENT':
            cmds = [f'HDG {acid} {int(value) if value else round(traf.trk[i])}']
        elif kind == 'SPEED':
            self.spd[i] = int(value) if value else p['spd']
            cmds = [f'SPD {acid} {self.spd[i]}']
        elif kind == 'CONTACT':
            uco.uco.toc(i, p['unit'])
            cmds = []
        elif kind == 'CLEARED':
            route = traf.ap.route[i]
            if self.distance(i, p['to']) is None:
                raise ValueError(f'{p["to"]} already passed')
            old = route.wpname[route.iactwp:route.wpname.index(p['to'], route.iactwp)]
            cmds = [f'{acid} BEFORE {p["to"]} ADDWPT {via}' for via in p['vias'] if via not in old]
            cmds += [f'DELWPT {acid} {wp}' for wp in old if wp not in p['vias']]
            cmds.append(f'DIRECT {acid} {p["vias"][0] if p["vias"] else p["to"]}')
        else:
            cmds = []
        stack.stack(*cmds)

    def reply(self, d, reply):
        i = traf.id2idx(d['acid'])
        d.update(state=reply, reply=reply, t_last=bs.sim.simt, t_timeout=None)
        if reply == 'STANDBY':
            return
        d.update(t_auto=None, auto_left=None)
        if reply == 'WILCO':
            if d['dtype'] == 'Level':
                self.cleared[i] = d['params']['level']
                self.cleared_ok[i] = [round(traf.selalt[i] / ft)]
                self.cross_level[i] = 0
            if d['kind'] == 'CONTACT':
                self.run(i, d) # hands over at WILCO
            if d['options']:
                self.schedule(d, EXEC_DELAY)

    def air_reply(self, d, reply):
        ''' Pilot answers a dialogue already closed on the ground, nothing gets flown '''
        d.update(air=reply, reply=reply)
        if reply != 'STANDBY':
            d.update(t_auto=None, auto_left=None)

    def execute(self, i, d, how, value=''):
        self.run(i, d, value if how == 'SET' else '')
        d.update(state='EXECUTED', exec=how if how == 'LOAD' else f'SET {value}'.strip(), t_auto=None, auto_left=None)

    @stack.command
    def dlup(self, acid: 'txt', cmd: 'txt', *args: 'txt'):
        ''' Uplink: DLUP acid cmd [values] unit, or DLUP ALL CHK unit for all of a unit's flights '''
        try:
            *args, unit = args
            if acid == 'ALL' and cmd == 'CHK':
                for i in range(traf.ntraf):
                    if self.logged_on[i] and uco.uco.uco[i] == unit:
                        try:
                            self.uplink(i, cmd, args, unit)
                        except ValueError:
                            pass
                return self.publish()
            i = traf.id2idx(acid)
            if i < 0:
                raise ValueError('Unknown flight')
            if not self.logged_on[i]:
                raise ValueError('Flight not logged on')
            if cmd != 'CHK' and uco.uco.uco[i] != unit:
                raise ValueError('Not UCO')
            self.uplink(i, cmd, args, unit)
        except ValueError as e:
            return self.error(str(e))
        return self.publish()

    def uplink(self, i, cmd, args, unit):
        d = self.message(i, cmd, args, unit)
        if any(o['acid'] == d['acid'] and o['dtype'] == d['dtype'] and o['state'] in OPEN for o in self.dialogues):
            raise ValueError('Dialogue of this type open')
        self.finish(d['acid'], d['dtype'])
        simt = bs.sim.simt
        d.update(id=len(self.dialogues) + 1, sender=unit, state='OPEN', was='', reply='', exec='',
                 t_sent=simt, t_last=simt, t_timeout=simt + TIMEOUT)
        self.dialogues.append(d)
        self.schedule(d, REPLY_DELAY)

    @stack.command
    def cross(self, acid: 'acid', level: 'txt', point: 'txt', cond: 'txt', cross: 'txt'):
        ''' Fly a crossing at once: CROSS acid 240 ARTIP ABV 180 (AT, ABV or BLW) '''
        try:
            self.run(acid, self.message(acid, 'EFL', [level, point, cond, cross], ''))
        except ValueError as e:
            return self.error(str(e))
        return True

    @stack.command
    def dlcancel(self, acid: 'acid', cmd: 'txt', value: 'txt' = ''):
        ''' Voice instruction given, replaces uplinks of its type. SPD 000 means route speeds again '''
        dtype = VOICE_TYPES.get(cmd)
        self.finish(traf.id[acid], dtype)
        if dtype == 'Level':
            self.cleared[acid] = 0
            if self.pending[acid]:
                self.pending[acid] = None
                stack.stack(f'VS {traf.id[acid]} 0')
        if dtype == 'Speed':
            self.spd[acid] = int(value or 0)
            if not self.spd[acid]:
                traf.swvnavspd[acid] = True
        return self.publish()

    @stack.command
    def dlview(self, acid: 'txt', on: 'onoff'):
        ''' Pilot panel opened (simulated pilot paused) or closed. The timeout keeps running '''
        simt = bs.sim.simt
        for d in self.dialogues:
            if d['acid'] != acid:
                continue
            if on and d['t_auto'] is not None:
                d.update(t_auto=None, auto_left=d['t_auto'] - simt)
            elif not on and d['auto_left'] is not None:
                d.update(t_auto=simt + d['auto_left'], auto_left=None)
        if on:
            self.viewed.add(acid)
        else:
            self.viewed.discard(acid)
        return True

    @stack.command
    def dlreply(self, dlg: int, reply: 'txt'):
        ''' Pilot panel: WILCO, UNABLE or STANDBY '''
        d = self.find(dlg)
        if d is None:
            return self.error('Unknown dialogue')
        if d.get('air') in OPEN and reply in REPLIES and reply != d['air']:
            self.air_reply(d, reply)
            return self.publish()
        if d['state'] == 'TIMEOUT':
            return self.error('Timed out: contact ATC by voice')
        if d['state'] not in OPEN or reply not in REPLIES or reply == d['state']:
            return self.error('Reply not possible')
        self.reply(d, reply)
        return self.publish()

    @stack.command
    def dlexec(self, dlg: int, how: 'txt', value: 'txt' = ''):
        ''' Pilot panel: LOAD, or SET with a value '''
        d = self.find(dlg)
        if d is None:
            return self.error('Unknown dialogue')
        if not (d['state'] == 'WILCO' or (d['state'] == 'EXECUTED' and how == 'SET')) or how not in d['options']:
            return self.error('Execution not possible')
        try:
            self.execute(traf.id2idx(d['acid']), d, how, value)
        except ValueError as e:
            return self.error(str(e))
        return self.publish()

    @stack.command
    def dlack(self, acid: 'txt'):
        ''' ACK: the controller acknowledges a flight's UNABLE and timed-out dialogues '''
        for d in self.dialogues:
            if d['acid'] == acid and d['state'] in AMBER:
                self.close(d)
        return self.publish()

    @stack.command
    def dlclose(self, acid: 'txt'):
        ''' CLOSE DLG: closes open dialogues on the ground only '''
        for d in self.dialogues:
            if d['acid'] == acid and d['state'] in OPEN:
                self.close(d)
                d['air'] = d['was'] # still open for the pilot
                self.schedule(d, REPLY_DELAY)
        return self.publish()

    @stack.command
    def dlcap(self, acid: 'acid', on: 'onoff'):
        ''' Whether a flight supports CPDLC; OFF also logs it off '''
        self.capable[acid] = on
        self.logged_on[acid] &= on
        return self.publish()

    @stack.command
    def dllogon(self, acid: 'acid', on: 'onoff'):
        ''' Whether a CPDLC-capable flight is logged on '''
        if on and not self.capable[acid]:
            return self.error('Flight not CPDLC capable')
        self.logged_on[acid] = on
        return self.publish()
