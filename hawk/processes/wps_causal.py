from pywps import Process, LiteralInput, LiteralOutput, UOM, ComplexInput, ComplexOutput  # noqa
from pywps.app.Common import Metadata
from pywps import FORMATS, Format
from pathlib import Path
import logging
import pandas as pd
from hawk.analysis import CausalAnalysis
import os

LOGGER = logging.getLogger("PYWPS")

FORMAT_PNG = Format("image/png", extension=".png", encoding="base64")
FORMAT_PDF = Format("application/pdf", extension=".pdf", encoding="utf-8")
FORMAT_PICKLE = Format("application/octet-stream", extension=".pkl")


class Causal(Process):
    def __init__(self):
        inputs = [
            ComplexInput(
                "dataset_train",
                "Train Dataset",
                abstract="Please add the train csv file here.",
                default="https://raw.githubusercontent.com/climateintelligence/hawk/main/hawk/demo/Ticino_train.csv",
                min_occurs=1,
                max_occurs=1,
                supported_formats=[FORMATS.CSV],
            ),
            ComplexInput(
                "dataset_test",
                "Test Dataset",
                abstract="Please add the test csv file here.",
                default="https://raw.githubusercontent.com/climateintelligence/hawk/main/hawk/demo/Ticino_test.csv",
                min_occurs=1,
                max_occurs=1,
                supported_formats=[FORMATS.CSV],
            ),
            LiteralInput(
                "target_column_name",
                "Target Column Name",
                data_type="string",
                abstract="Please enter the case-specific name of the target variable in the dataframe.",
                default="target",
            ),
            LiteralInput(
                "pcmci_test_choice",
                "PCMCI Test Choice",
                data_type="string",
                abstract="Choose the independence test to be used in PCMCI.",
                default="ParCorr",
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
                default="1",
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
                default="both",
                allowed_values=[
                    "forward",
                    "backward",
                    "both",
                ],
            ),
            LiteralInput(
                "tefs_use_contemporary_features",
                "TEFS Use Contemporary Features",
                data_type="boolean",
                abstract="Choose whether to use contemporary features in the TEFS algorithm.",
                default=True,
            ),
            LiteralInput(
                "tefs_max_lag_features",
                "TEFS Max Lag Features",
                data_type="string",
                abstract="Choose the maximum lag of the features in the TEFS algorithm.",
                default="1",
                allowed_values=[
                    "no_lag",
                    "1",
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
                default="1",
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
            ComplexOutput(
                "pkl_baseline",
                "Baseline Scores",
                abstract="The baseline scores on the initial data.",
                as_reference=True,
                supported_formats=[FORMAT_PICKLE],
            ),
            ComplexOutput(
                "plot_pcmci",
                "Selected features by PCMCI",
                abstract="The selected features by PCMCI.",
                as_reference=True,
                supported_formats=[FORMAT_PDF],
            ),
            ComplexOutput(
                "plot_pcmci_preview",
                "Selected features by PCMCI",
                abstract="The selected features by PCMCI.",
                as_reference=True,
                supported_formats=[FORMAT_PNG],
            ),
            ComplexOutput(
                "pkl_pcmci",
                "PCMCI Results Details",
                abstract="The PCMCI results details.",
                as_reference=True,
                supported_formats=[FORMAT_PICKLE],
            ),
            ComplexOutput(
                "plot_tefs",
                "Selected features by TEFS",
                abstract="The selected features by TEFS.",
                as_reference=True,
                supported_formats=[FORMAT_PDF],
            ),
            ComplexOutput(
                "plot_tefs_preview",
                "Selected features by TEFS",
                abstract="The selected features by TEFS.",
                as_reference=True,
                supported_formats=[FORMAT_PNG],
            ),
            ComplexOutput(
                "pkl_tefs",
                "TEFS Results",
                abstract="The TEFS results.",
                as_reference=True,
                supported_formats=[FORMAT_PICKLE],
            ),
            ComplexOutput(
                "plot_tefs_wrapper",
                "Wrapper scores by TEFS",
                abstract="The wrapper scores evolution by TEFS.",
                as_reference=True,
                supported_formats=[FORMAT_PDF],
            ),
            ComplexOutput(
                "plot_tefs_wrapper_preview",
                "Wrapper scores by TEFS",
                abstract="The wrapper scores evolution by TEFS.",
                as_reference=True,
                supported_formats=[FORMAT_PNG],
            ),
            ComplexOutput(
                "pkl_tefs_wrapper",
                "TEFS Wrapper Scores Evolution details",
                abstract="The TEFS wrapper scores evolution details.",
                as_reference=True,
                supported_formats=[FORMAT_PICKLE],
            ),
        ]

        super(Causal, self).__init__(
            self._handler,
            identifier="causal",
            title="Causal Analysis",
            abstract="Performs a causal analysis with multiple configurations, returning outputs plots and pkl file.",
            keywords=["causal", "analysis"],
            metadata=[
                Metadata("PyWPS", "https://pywps.org/"),
                Metadata("Birdhouse", "http://bird-house.github.io/"),
                Metadata("PyWPS Demo", "https://pywps-demo.readthedocs.io/en/latest/"),
                Metadata("Emu: PyWPS examples", "https://emu.readthedocs.io/en/latest/"),
            ],
            version="1.5",
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True,
        )

    def _handler(self, request, response):
        response.update_status("Processing started", 0)

        # Read the inputs
        target_column_name = request.inputs["target_column_name"][0].data

        df_train = pd.read_csv(request.inputs["dataset_train"][0].file, header=0)
        df_test = pd.read_csv(request.inputs["dataset_test"][0].file, header=0)

        LOGGER.info(f"Train shape: {df_train.shape}")
        LOGGER.info(f"Test shape: {df_test.shape}")
        LOGGER.info(f"Train columns: {df_train.columns}")
        LOGGER.info(f"Test columns: {df_test.columns}")

        pcmci_test_choice = request.inputs["pcmci_test_choice"][0].data
        pcmci_max_lag = int(request.inputs["pcmci_max_lag"][0].data)

        tefs_direction = request.inputs["tefs_direction"][0].data
        tefs_use_contemporary_features = request.inputs["tefs_use_contemporary_features"][0].data
        if str(request.inputs["tefs_max_lag_features"][0].data) == "no_lag":
            tefs_max_lag_features = 0
        else:
            tefs_max_lag_features = int(request.inputs["tefs_max_lag_features"][0].data)
        tefs_max_lag_target = int(request.inputs["tefs_max_lag_target"][0].data)

        workdir = Path(self.workdir)
        os.environ['MPLCONFIGDIR'] = os.path.join(workdir, "/matplotlib")

        if not tefs_use_contemporary_features and tefs_max_lag_features == 0:
            raise ValueError("You cannot use no lag features and not use contemporary features in TEFS.")

        causal_analysis = CausalAnalysis(
            df_train,
            df_test,
            target_column_name,
            pcmci_test_choice,
            pcmci_max_lag,
            tefs_direction,
            tefs_use_contemporary_features,
            tefs_max_lag_features,
            tefs_max_lag_target,
            workdir,
            response,
        )

        causal_analysis.run()

        response.outputs["pkl_baseline"].file = causal_analysis.baseline
        response.outputs["plot_pcmci"].file = causal_analysis.plot_pcmci["pdf"]
        response.outputs["pkl_pcmci"].file = causal_analysis.details_pcmci
        response.outputs["plot_tefs"].file = causal_analysis.plot_tefs["pdf"]
        response.outputs["pkl_tefs"].file = causal_analysis.details_tefs
        response.outputs["plot_tefs_wrapper"].file = causal_analysis.plot_tefs_wrapper["pdf"]
        response.outputs["pkl_tefs_wrapper"].file = causal_analysis.details_tefs_wrapper

        # Previews for the plots in png format
        response.outputs["plot_pcmci_preview"].file = causal_analysis.plot_pcmci["png"]
        response.outputs["plot_tefs_preview"].file = causal_analysis.plot_tefs["png"]
        response.outputs["plot_tefs_wrapper_preview"].file = causal_analysis.plot_tefs_wrapper["png"]

        response.update_status("Processing completed", 100)

        return response
