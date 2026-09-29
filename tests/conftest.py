import pytest

from decisionforge.data.store import DataStore
from decisionforge.data.synth import generate
from decisionforge.llm import OfflineLLM


@pytest.fixture(scope="session")
def df():
    return generate()


@pytest.fixture(scope="session")
def store(df):
    return DataStore(df)


@pytest.fixture()
def llm():
    return OfflineLLM()
