from pywps import Process, LiteralInput, LiteralOutput, UOM, ComplexInput, ComplexOutput
from pywps.app.Common import Metadata
from pywps import FORMATS, Format
from pathlib import Path
import logging

LOGGER = logging.getLogger("PYWPS")


class Causal(Process):
    """A nice process saying 'hello'."""

    def __init__(self):
        inputs = [
            LiteralInput(
                "target_column_name",
                "Target Column Name",
                data_type="string",
                abstract="Please enter the case-specific name of the target variable in the dataframe.",
            ),
            LiteralInput(
                "pcmci_test_choice",
                "PCMCI Test Choice",
                data_type="string",
                abstract="Choose the independence test to be used in PCMCI.",
                allowed_values=[
                    "ParCorr",
                    "CMIknn",
                ],
            ),
            LiteralInput(
                "pcmci_max_lag",
                "PCMCI Max Lag",
                data_type="string",
                abstract="Choose the maximum lag to test used in PCMCI.",
                allowed_values=[
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                ],
            ),
            LiteralInput(
                "tefs_direction",
                "TEFS Direction",
                data_type="string",
                abstract="Choose the direction of the TEFS algorithm.",
                allowed_values=[
                    "forward",
                    "backward",
                    "both",
                ],
            ),
            LiteralInput(
                "tefs_use_comtemporary_features",
                "TEFS Use Comtemporary Features",
                data_type="boolean",
                abstract="Choose whether to use comtemporary features in the TEFS algorithm.",
                default=False,
            ),
            LiteralInput(
                "tefs_max_lag_features",
                "TEFS Max Lag Features",
                data_type="string",
                abstract="Choose the maximum lag of the features in the TEFS algorithm.",
                allowed_values=[
                    "no_lag" "1",
                    "2",
                    "3",
                    "4",
                    "5",
                ],
            ),
            LiteralInput(
                "tefs_max_lag_target",
                "TEFS Max Lag Target",
                data_type="string",
                abstract="Choose the maximum lag of the target in the TEFS algorithm.",
                allowed_values=[
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                ],
            ),
        ]
        outputs = [
            LiteralOutput(
                "output",
                "Output response",
                abstract="A friendly Hello from us.",
                keywords=["output", "result", "response"],
                data_type="string",
            )
        ]

        super(Causal, self).__init__(
            self._handler,
            identifier="hello",
            title="Say Hello",
            abstract="Just says a friendly Hello."
            "Returns a literal string output with Hello plus the inputed name.",
            keywords=["hello", "demo"],
            metadata=[
                Metadata("PyWPS", "https://pywps.org/"),
                Metadata("Birdhouse", "http://bird-house.github.io/"),
                Metadata("PyWPS Demo", "https://pywps-demo.readthedocs.io/en/latest/"),
                Metadata(
                    "Emu: PyWPS examples", "https://emu.readthedocs.io/en/latest/"
                ),
            ],
            version="1.5",
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True,
        )

    def _handler(self, request, response):
        response.update_status("Processing started", 0)

        # read the respons
        target_column_name = request.inputs["target_column_name"][0].data
        pcmci_test_choice = request.inputs["pcmci_test_choice"][0].data
        pcmci_max_lag = request.inputs["pcmci_max_lag"][0].data
        tefs_direction = request.inputs["tefs_direction"][0].data
        tefs_use_comtemporary_features = request.inputs[
            "tefs_use_comtemporary_features"
        ][0].data
        tefs_max_lag_features = request.inputs["tefs_max_lag_features"][0].data
        tefs_max_lag_target = request.inputs["tefs_max_lag_target"][0].data

        workdir = Path(self.workdir)

        # connect to the analysis class

        response.outputs["output"].data = "Hello " + request.inputs["name"][0].data
        response.outputs["output"].uom = UOM("unity")
        return response
