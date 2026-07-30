from contextlib import contextmanager


def create_cuda_stream_pair(device, enabled=None):
    return {}, {}


@contextmanager
def cuda_stream_context(stream_name, streams_dict, events_dict, device, enabled=None):
    yield


def synchronize_cuda_streams(events_dict, device, enabled=None):
    return None
