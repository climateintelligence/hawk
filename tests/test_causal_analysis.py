#!/usr/bin/env python

"""Tests for `hawk` package."""

import os

import pandas as pd
import pytest
from click.testing import CliRunner  # noqa: F401

import hawk  # noqa: F401
from hawk import cli  # noqa: F401
from hawk.analysis import CausalAnalysis


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_causal_analysis():
    df_train = pd.read_csv("hawk/demo/Ticino_train.csv", header=0)
    df_test = pd.read_csv("hawk/demo/Ticino_test.csv", header=0)
    target_column_name = "target"
    pcmci_test_choice = "ParCorr"
    pcmci_max_lag = 2
    tefs_direction = "both"
    tefs_use_contemporary_features = True
    tefs_max_lag_features = 1
    tefs_max_lag_target = 1
    workdir = "tests/output"

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
        response=None,
    )

    causal_analysis.run()

    os.system("rm -r tests/output")
