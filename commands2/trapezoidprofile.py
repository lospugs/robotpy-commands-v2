# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
from __future__ import annotations

from typing import TypeVar, Generic, Protocol, overload, ClassVar
import math
import wpimath.units
from wpimath.trajectory import TrapezoidProfileRadians

TS = TypeVar("TS")  # Profile.State
TC = TypeVar("TC")  # Profile.Constraints


class TrapezoidProfile(Protocol, Generic[TS, TC]):
    """
    A trapezoid-shaped velocity profile.

    While this class can be used for a profiled movement from start to finish,
    the intended usage is to filter a reference's dynamics based on trapezoidal
    velocity constraints. To compute the reference obeying this constraint, do
    the following.

    Initialization::

      constraints = TrapezoidProfile.Constraints(kMaxV, kMaxA)
      previousProfiledReference = initialReference

    Run on update::

      profile = TrapezoidProfile(constraints, unprofiledReference, previousProfiledReference)
      previousProfiledReference = profile.calculate(timeSincePreviousUpdate)

    where ``unprofiledReference`` is free to change between calls. Note that
    when the unprofiled reference is within the constraints,
    :meth:`calculate` returns the unprofiled reference unchanged.

    Otherwise, a timer can be started to provide monotonic values for
    ``calculate()`` and to determine when the profile has completed via
    :meth:`isFinished`.
    """

    _endDeccel: wpimath.units.seconds
    _endAccel: wpimath.units.seconds
    _endFullSpeed: float
    _direction: int
    _constraints: TC
    _current: TS
    _goal: TS

    class Constraints:
        def __init__(
            self,
            maxVelocity: wpimath.units.units_per_second = 0,
            maxAcceleration: wpimath.units.units_per_second_squared = 0,
        ) -> None:
            self._maxAcceleration = maxAcceleration
            self._maxVelocity = maxVelocity

        @property
        def maxAcceleration(self) -> wpimath.units.units_per_second_squared:
            return self._maxAcceleration

        @property
        def maxVelocity(self) -> wpimath.units.units_per_second:
            return self._maxVelocity

    class State:
        __hash__: ClassVar[None] = None
        position: float
        velocity: wpimath.units.units_per_second

        def __eq__(self, arg0: TS) -> bool:
            return (self.position == arg0.position) and (self.velocity == arg0.velocity)

        def __init__(
            self, position: float = 0, velocity: wpimath.units.units_per_second = 0
        ) -> None:
            self.position = position
            self.velocity = velocity

        def __repr__(self) -> str:
            return f"State - pos, vel"

    @overload
    def __init__(self, constraints: TC) -> None:
        """
        Construct a TrapezoidProfile.

        :param constraints: The constraints on the profile, like maximum velocity.
        """
        ...

    def __init__(self, constraints: TC, goal: TS = None, initial: TS = None) -> None:
        """
        Construct a TrapezoidProfile.

        :deprecated: Pass the desired and current state into calculate instead of
                     constructing a new TrapezoidProfile with the desired and current state

        :param constraints: The constraints on the profile, like maximum velocity.
        :param goal:        The desired state when the profile is complete.
        :param initial:     The initial state (usually the current state).
        """
        if initial is None and goal is None:
            # this is the old constructor
            self._constraints = constraints
            return

        # This is the new constructor style
        self._direction = -1 if self._shouldFlipAcceleration(initial, goal) else 1
        self._current = self._direct(initial)
        self._direct(goal)

        if self._current.velocity > self._constraints.maxVelocity:
            self._current.velocity = self._constraints.maxVelocity

        # Deal with a possibly truncated motion profile (with nonzero initial or
        # final velocity) by calculating the parameters as if the profile began and
        # ended at zero velocity
        cutoffBegin = self._current.velocity / self._constraints.maxAcceleration
        cutoffDistBegin = (
            cutoffBegin * cutoffBegin * self._constraints.maxAcceleration / 2.0
        )

        cutoffEnd = goal.velocity / self._constraints.maxAcceleration
        cutoffDistEnd = cutoffEnd * cutoffEnd * self._constraints.maxAcceleration / 2.0

        # Now we can calculate the parameters as if it was a full trapezoid instead
        # of a truncated one

        fullTrapezoidDist = (
            cutoffDistBegin + (goal.position - self._current.position) + cutoffDistEnd
        )
        accelerationTime = (
            self._constraints.maxVelocity / self._constraints.maxAcceleration
        )

        fullSpeedDist = (
            fullTrapezoidDist
            - accelerationTime * accelerationTime * self._constraints.maxAcceleration
        )

        # Handle the case where the profile never reaches full speed
        if fullSpeedDist < 0:
            accelerationTime = math.sqrt(
                fullTrapezoidDist / self._constraints.maxAcceleration
            )
            fullSpeedDist = 0

        self._endAccel = accelerationTime - cutoffBegin
        self._endFullSpeed = (
            self._endAccel + fullSpeedDist / self._constraints.maxVelocity
        )
        self._endDeccel = self._endFullSpeed + accelerationTime - cutoffEnd
        result = TS(self._current.position, self._current.velocity)

    def calculate(self, t: wpimath.units.seconds, goal: TS, current: TS) -> TS:
        """
        Calculate the correct position and velocity for the profile at a time t
        where the beginning of the profile was at time t = 0.

        :param t:       The time since the beginning of the profile.
        :param goal:    The desired state when the profile is complete.
        :param current: The initial state (usually the current state).
        """
        self._direction = -1 if self._shouldFlipAcceleration(current, goal) else 1
        self._direct(current)
        self._current = current
        self._direct(goal)

        if self._current.velocity > self._constraints.maxVelocity:
            self._current.velocity = self._constraints.maxVelocity

        # Deal with a possibly truncated motion profile (with nonzero initial or
        # final velocity) by calculating the parameters as if the profile began and
        # ended at zero velocity
        cutoffBegin = self._current.velocity / self._constraints.maxAcceleration
        cutoffDistBegin = (
            cutoffBegin * cutoffBegin * self._constraints.maxAcceleration / 2.0
        )

        cutoffEnd = goal.velocity / self._constraints.maxAcceleration
        cutoffDistEnd = cutoffEnd * cutoffEnd * self._constraints.maxAcceleration / 2.0

        # Now we can calculate the parameters as if it was a full trapezoid instead
        # of a truncated one

        fullTrapezoidDist = (
            cutoffDistBegin + (goal.position - self._current.position) + cutoffDistEnd
        )
        accelerationTime = (
            self._constraints.maxVelocity / self._constraints.maxAcceleration
        )

        fullSpeedDist = (
            fullTrapezoidDist
            - accelerationTime * accelerationTime * self._constraints.maxAcceleration
        )

        # Handle the case where the profile never reaches full speed
        if fullSpeedDist < 0:
            accelerationTime = math.sqrt(
                fullTrapezoidDist / self._constraints.maxAcceleration
            )
            fullSpeedDist = 0

        self._endAccel = accelerationTime - cutoffBegin
        self._endFullSpeed = (
            self._endAccel + fullSpeedDist / self._constraints.maxVelocity
        )
        self._endDeccel = self._endFullSpeed + accelerationTime - cutoffEnd

        if isinstance(self._current, TrapezoidProfile.State):
            result = TrapezoidProfile.State(
                self._current.position, self._current.velocity
            )
        elif isinstance(self._current, TrapezoidProfileRadians.State):
            result = TrapezoidProfileRadians.State(
                self._current.position, self._current.velocity
            )

        if t < self._endAccel:
            result.velocity += t * self._constraints.maxAcceleration
            result.position += (
                self._current.velocity + t * self._constraints.maxAcceleration / 2.0
            ) * t
        elif t < self._endFullSpeed:
            result.velocity = self._constraints.maxVelocity
            result.position += (
                self._current.velocity
                + self._endAccel * self._constraints.maxAcceleration / 2.0
            ) * self._endAccel + self._constraints.maxVelocity * (t - self._endAccel)
        elif t <= self._endDeccel:
            result.velocity = (
                goal.velocity
                + (self._endDeccel - t) * self._constraints.maxAcceleration
            )
            timeLeft = self._endDeccel - t
            result.position = (
                goal.position
                - (goal.velocity + timeLeft * self._constraints.maxAcceleration / 2.0)
                * timeLeft
            )
        else:
            result = goal

        self._direct(result)

        return result

    def isFinished(self, t: wpimath.units.seconds) -> bool:
        """
        Returns true if the profile has reached the goal.

        The profile has reached the goal if the time since the profile started
        has exceeded the profile's total time.

        :param t: The time since the beginning of the profile.
        """
        return t >= self.totalTime()

    def timeLeftUntil(self, target: float) -> wpimath.units.seconds:
        """
        Returns the time left until a target distance in the profile is reached.

        :param target: The target distance.
        """
        position = self._current.position * self._direction
        velocity = self._current.velocity * self._direction
        endAccel = self._endAccel * self._direction
        endFullSpeed = self._endFullSpeed * self._direction - endAccel

        if target < position:
            endAccel *= -1
            endFullSpeed *= -1
            velocity *= -1

        endAccel = max(endAccel, 0)
        endFullSpeed = max(endFullSpeed, 0)

        acceleration = self._constraints.maxAcceleration
        deceleration = acceleration * -1

        distToTarget = abs(target - position)
        if distToTarget < 0.000001:
            return 0

        accelDist = velocity * endAccel + 0.5 * acceleration * endAccel * endAccel

        deccelVelocity = (
            velocity
            if endAccel > 0
            else math.sqrt(abs(velocity * velocity + 2 * acceleration * accelDist))
        )

        fullSpeedDistance = self._constraints.maxVelocity * endFullSpeed

        decelDist = 0
        if accelDist > distToTarget:
            accelDist = distToTarget
            fullSpeedDistance = 0
            decelDist = 0
        elif (accelDist + fullSpeedDistance) > distToTarget:
            fullSpeedDistance = distToTarget - accelDist
            decelDist = 0
        else:
            decelDist = distToTarget - fullSpeedDistance - accelDist

        accelTime = (
            -velocity + math.sqrt(velocity * velocity + 2 * acceleration * accelDist)
        ) / acceleration
        deccelTime = (
            -deccelVelocity
            + math.sqrt(
                abs(deccelVelocity * deccelVelocity + 2 * deceleration * decelDist)
            )
        ) / deceleration

        fullSpeedTime = fullSpeedDistance / self._constraints.maxVelocity

        return accelTime + fullSpeedTime + deccelTime

    def totalTime(self) -> wpimath.units.seconds:
        """
        Returns the total time the profile takes to reach the goal.
        """
        return self._endDeccel

    def _direct(self, inbound: TS) -> None:
        """
        If the current profile's direction is negative, flip the state variables
        to match.

        :param  The State to modify
        """
        inbound.position *= self._direction
        inbound.velocity *= self._direction

    def _shouldFlipAcceleration(self, initial: TS, goal: TS) -> bool:
        """Returns true if the profile inverted.

        The profile is inverted if goal position is less than the initial position.

        :param initial: The initial state (usually the current state).
        :param goal:    The desired state when the profile is complete.
        """
        return initial.position > goal.position
