from .wps_causal import Causal
from .wps_say_hello import SayHello

processes = [
    SayHello(),
    Causal(),
]
