#!/usr/bin/env python

"""Tests for `hawk` package."""

import pytest

from click.testing import CliRunner  # noqa: F401

import hawk  # noqa: F401
from hawk import cli  # noqa: F401


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
