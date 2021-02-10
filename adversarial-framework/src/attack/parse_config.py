
def map_objective(name):
    """

    :param name: str
    :param evasion: EvasionMethod to be added
    :return:
    """
    from src import attack
    cls = getattr(attack, name)
    return cls()

# def load_attacks(attack_file_name):
#     with open(attack_file_name) as stream:
#         yaml = YAML(typ='safe')
#         attacks = yaml.load(stream)
#
#     # Many-to-many cartesian product
#     objectives = attacks['objectives']
#     evasions = attacks['evasion']
#     backdoors = attacks['backdoors']
#
#     return attacks

# class AttackConfig():


