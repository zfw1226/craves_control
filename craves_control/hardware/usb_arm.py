"""Maplin USB Robot arm control"""
import os

os.environ['DYLD_LIBRARY_PATH'] = '/opt/local/lib'
import usb.core
from time import sleep
import numpy as np
import time

class BitPattern(object):
    """A bit pattern to send to a robot arm"""
    __slots__ = ['arm', 'base', 'led']

    def __init__(self, arm, base, led):
        self.arm = arm
        self.base = base
        self.led = led

    def __iter__(self):
        return iter([self.arm, self.base, self.led])

    def __getitem__(self, item):
        return [self.arm, self.base, self.led][item]

    def __or__(self, other):
        return BitPattern(self.arm | other.arm,
                          self.base | other.base,
                          self.led | other.led)

    def __eq__(self, other):
        return self.arm == other.arm and self.base == other.base and self.led == other.led

    def __repr__(self):
        return "<BitPattern arm:%s base:%s led:%s>" % (self.arm, self.base, self.led)

    def __str__(self):
        return self.__repr__()


GripsClose = BitPattern(1, 0, 0)
CloseGrips = GripsClose
GripsOpen = BitPattern(2, 0, 0)
OpenGrips = GripsOpen
Stop = BitPattern(0, 0, 0)
WristUp = BitPattern(0x4, 0, 0)
WristDown = BitPattern(0x8, 0, 0)
ElbowUp = BitPattern(0x10, 0, 0)
ElbowDown = BitPattern(0x20, 0, 0)
ShoulderUp = BitPattern(0x40, 0, 0)
ShoulderDown = BitPattern(0x80, 0, 0)
BaseClockWise = BitPattern(0, 1, 0)
BaseCtrClockWise = BitPattern(0, 2, 0)
LedOn = BitPattern(0, 0, 1)

pattern = [BaseClockWise, BaseCtrClockWise,
           ShoulderDown, ShoulderUp,
           ElbowDown, ElbowUp,
           WristDown, WristUp
           ]


class Arm(object):
    """Arm interface"""
    __slots__ = ['dev']

    def __init__(self):
        self.dev = usb.core.find(idVendor=0x1267)
        self.dev.set_configuration()

    def tell(self, msg):
        """Send a USB messaqe to the arm"""
        self.dev.ctrl_transfer(0x40, 6, 0x100, 0, msg)

    def safe_tell(self, fn):
        """Send a message to the arm, with a stop
        to ensure that the robot stops in the
        case of an exception"""
        try:
            fn()
        except:
            self.tell(Stop)
            raise

    def ctl(self, vector, time=0.1):
        full_patern = LedOn
        for i, info in enumerate(vector):
            if info != 0:
                full_patern |= pattern[i * 2 + max(info, 0)]

                # print pattern[i*2+max(info,0)]
        self.tell(full_patern)
        sleep(time)
        self.tell(Stop)

    def tell_ctl(self, vector):
        full_patern = Stop
        for i, info in enumerate(vector):
            if info != 0:
                full_patern |= pattern[i * 2 + max(int(info), 0)]

                # print pattern[i*2+max(info,0)]
        self.tell(full_patern)

    def pwm_ctl(self, vector, t=0.1):
        dircetion = np.ones(len(vector))
        dircetion[np.where(np.array(vector) < 0)] = -1
        vector[0] *= 2
        if vector[1] > 0:
            vector[1] *= 4
        if vector[2] > 0:
            vector[2] *= 2
        vector = np.abs(vector)*t

        t_max = np.max(vector)
        t0 = time.time()
        while True:
            dt = time.time() - t0
            cmd = np.zeros(len(vector))
            if dt > t_max:
                self.tell(Stop)
                break
            cmd[np.where(dt < vector)] = 1
            cmd = cmd * dircetion
            self.tell_ctl(cmd)

    def grip_open(self, t=1):
        self.tell(GripsOpen)
        sleep(t)
        self.tell(Stop)

    def grip_close(self, t=1):
        self.tell(CloseGrips)
        sleep(t)
        self.tell(Stop)

    def stop(self):
        self.tell_ctl(Stop)

    def move(self, pattern, time=1):
        """Perform a pattern move with timing and stop"""
        self.tell(pattern)
        sleep(time)
        self.tell(Stop)

    def doActions(self, actions):
        """Params: List of actions - each is a list/tuple of BitPattern and time
         (defaulting to 1 if not set)"""
        # Validate
        for action in actions:
            if not 1 <= len(action) <= 2:
                raise ValueError("Wrong number of parameters in action %s" %
                                 (repr(action)))
            if not isinstance(action[0], BitPattern):
                raise ValueError("Not a valid action")
        # Do
        try:
            for action in actions:
                if len(action) == 2:
                    time = action[1]
                else:
                    time = 1
                self.move(action[0], time)
        except:
            self.move(Stop)
            raise


block_left = [[ShoulderDown], [GripsClose, 0.4], [ShoulderUp],
              [BaseClockWise, 10.2], [ShoulderDown],
              [GripsOpen, 0.4], [ShoulderUp, 1.2]]
block_right = [[ShoulderDown], [GripsClose, 0.4], [ShoulderUp],
               [BaseCtrClockWise, 10.2], [ShoulderDown],
               [GripsOpen, 0.4], [ShoulderUp, 1.2]]
left_and_blink = list(block_left)
left_and_blink.extend([[LedOn, 0.5], [Stop, 0.5]] * 3)