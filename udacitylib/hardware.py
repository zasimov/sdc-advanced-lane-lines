"""udacitylib.hardware module

Module contains functions to get information about hardware.

"""

from tensorflow.python.client import device_lib


def get_available_gpus():
    """get_available_gpus returns a list of names of GPU devices 
    
    Source https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    gpus = get_available_gpus()
    print('GPU: %s' % gpus)
    if not gpus:
        exit(1)
