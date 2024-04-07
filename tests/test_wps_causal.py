from pywps import Service  # noqa: F401
from pywps.tests import assert_response_success, client_for  # noqa: F401

from hawk.processes.wps_causal import Causal  # noqa: F401

from .common import get_output  # noqa: F401


# def test_wps_causal():
#     client = client_for(Service(processes=[Causal()]))
#     datainputs = "name=LovelySugarBird"
#     resp = client.get(
#         "?service=WPS&request=Execute&version=1.0.0&identifier=causal&datainputs={}".format(
#             datainputs))
#     assert_response_success(resp)
#     assert get_output(resp.xml) == {'output': "Hello LovelySugarBird"}
