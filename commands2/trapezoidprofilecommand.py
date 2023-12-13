# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
from __future__ import annotations

from typing import TypeVar, Callable, Any, Generic

from wpilib import Timer
from wpimath.trajectory import TrapezoidProfile, TrapezoidProfileRadians

from .command import Command
from .subsystem import Subsystem

# Defined two generic types for the Profile and ProfileState variables.
# This allows an implementation for both dimensionless and Radians
# instances of TrapezoidProfiles (or any future dimension as well)
TS = TypeVar("TS")  # Generic[TrapezoidProfile.State]


class TrapezoidProfileCommand(Command, Generic[TS]):
    """
    A command that runs a :class:`.TrapezoidProfile`. Useful for smoothly controlling mechanism motion.
    """

    def __init__(
        self,
        profile: TrapezoidProfile,
        output: Callable[[TS], Any],
        getGoal: Callable[[], TS],
        getCurrent: Callable[[], TS],
        *requirements: Subsystem,
    ):
        """Creates a new TrapezoidProfileCommand that will execute the given :class:`.TrapezoidProfile`.
        Output will be piped to the provided consumer function.

        :param profile:      The motion profile to execute.
        :param output:       The consumer for the profile output.
        :param getGoal:      The supplier for the desired state
        :param getCurrent:   The supplier for the current state
        :param requirements: The subsystems required by this command.
        """
        super().__init__()
        self._profile = profile
        self._output = output
        self._getGoal = getGoal
        self._getCurrent = getCurrent
        self._timer = Timer()

        self.addRequirements(*requirements)

    def initialize(self) -> None:
        self._timer.restart()

    def execute(self) -> None:
        self._output(
            self._profile.calculate(  # type: ignore[attr-defined]
                self._timer.get(), self._getGoal(), self._getCurrent()
            )
        )

    def end(self, interrupted) -> None:
        self._timer.stop()

    def isFinished(self) -> bool:
        return self._timer.hasElapsed(self._profile.totalTime())  # type: ignore[attr-defined]
