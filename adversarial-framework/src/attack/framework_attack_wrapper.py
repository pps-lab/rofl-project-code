

class FrameworkAttackWrapper(object):
    """Wraps an attack with dict params to invocate later."""

    def __init__(self, attack, kwargs):
        self.attack = attack
        self.kwargs = kwargs