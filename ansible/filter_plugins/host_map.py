


def host_map(value, hosts, num_clients_per_host):
    """
    Calculates client host assignment
    """
    print(hosts)
    return {
        'key': value,
        'value': hosts[(value // int(num_clients_per_host)) % len(hosts)]
    }

class FilterModule(object):
    ''' jinja2 filters '''

    def filters(self):
        return {
            'host_map': host_map,
        }