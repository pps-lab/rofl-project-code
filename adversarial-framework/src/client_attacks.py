from enum import Enum

class Attack(Enum):

    # Targeted attack strategies
    BACKDOOR = "backdoor"

    # Byzantine strategies
    UNTARGETED = 'untargeted'
    DEVIATE_MAX_NORM = 'deviate_max_norm'

    @staticmethod
    def is_targeted(type) -> bool:
        """

        :type type: Attack
        """
        if type == Attack.BACKDOOR:
            return True
        else:
            return False
