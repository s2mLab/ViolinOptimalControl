from bioptim import MichaudFatigue, EffortPerception, MichaudTauFatigue, TauEffortPerception


class FatigueModels:
    def __init__(self, violin, max_target):
        self.XIA = MichaudFatigue(
            **violin.fatigue_parameters(
                fatigue_type=MichaudTauFatigue,
                direction=1,
                index=0,
                effort_threshold=0,
                stabilization_factor=0,
                effort_factor=0,
                scaling=max_target,
            )
        )

        self.XIA_STABILIZED = MichaudFatigue(
            **violin.fatigue_parameters(
                fatigue_type=MichaudTauFatigue,
                direction=1,
                index=0,
                effort_threshold=0,
                effort_factor=0,
                scaling=max_target,
            )
        )

        self.MICHAUD = MichaudFatigue(
            **violin.fatigue_parameters(
                fatigue_type=MichaudTauFatigue,
                direction=1,
                index=0,
            )
        )

        self.EFFORT_PERCEPTION = EffortPerception(
            **violin.fatigue_parameters(
                fatigue_type=TauEffortPerception,
                direction=1,
                index=0,
            )
        )
