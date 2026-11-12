from contextlib import contextmanager


@contextmanager
def hold_out_net(network):
    params = [p for p in network.parameters() if p.requires_grad]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p in params:
            p.requires_grad_(True)
