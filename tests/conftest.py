import pytest
import numpy as np
import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict


class TestSettings(BaseSettings):
    """Centralized test environment configuration.

    Attributes:
        SEED (int): seed for reproduciblity
    """

    SEED: int = 17

    model_config = SettingsConfigDict(env_file=".env.test", frozen=True, extra="forbid")


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    """
    Provides the test settings configuration.

    Returns:
        TestSettings: Loaded configuration object

    Raises:
        pytest.Exception: On invalid/missing configuration
    """
    try:
        return TestSettings()
    except Exception as e:
        pytest.fail(f"Test configuration failed: {str(e)}")


@pytest.fixture
def sample_dataframe(test_settings: TestSettings) -> pd.DataFrame:
    """
    Provides a sample DataFrame for testing purposes.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        pd.DataFrame: Sample DataFrame with random data.
    """
    np.random.seed(test_settings.SEED)
    return pd.DataFrame(
        {
            "A": np.random.rand(100),
            "B": np.random.rand(100),
            "C": np.random.randint(0, 10, size=100),
            "D": np.random.choice(["X", "Y", "Z"], size=100),
        }
    )


@pytest.fixture
def sample_series(test_settings: TestSettings) -> pd.Series:
    """
    Provides a sample Series for testing purposes.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        pd.Series: Sample Series with random data.
    """
    np.random.seed(test_settings.SEED)
    return pd.Series(np.random.rand(50), name="test_series")


@pytest.fixture
def sample_numpy_array(test_settings: TestSettings) -> np.ndarray:
    """
    Provides a sample numpy array for testing purposes.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        np.ndarray: Sample numpy array with random data.
    """
    np.random.seed(test_settings.SEED)
    return np.random.rand(30)


@pytest.fixture
def sample_list(test_settings: TestSettings) -> list[float]:
    """
    Provides a sample list for testing purposes.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        list[float]: Sample list with random data.
    """
    np.random.seed(test_settings.SEED)
    return np.random.rand(20).tolist()


@pytest.fixture
def correlation_dataframe(test_settings: TestSettings) -> pd.DataFrame:
    """
    Provides a correlation matrix DataFrame for heatmap testing.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        pd.DataFrame: Correlation matrix with values between -1 and 1.
    """
    np.random.seed(test_settings.SEED)
    data = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
        }
    )
    return data.corr()


@pytest.fixture
def missing_values_data() -> list[int]:
    """
    Provides sample missing values count data for bar plot testing.

    Returns:
        list[int]: List of missing value counts.
    """
    return [5, 12, 3, 8, 15, 0, 2]


@pytest.fixture
def feature_names() -> list[str]:
    """
    Provides sample feature names for testing.

    Returns:
        list[str]: List of feature names.
    """
    return [
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
    ]


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """
    Provides an empty DataFrame for edge case testing.

    Returns:
        pd.DataFrame: Empty DataFrame.
    """
    return pd.DataFrame()


@pytest.fixture
def single_value_dataframe() -> pd.DataFrame:
    """
    Provides a DataFrame with a single value for edge case testing.

    Returns:
        pd.DataFrame: DataFrame with single value.
    """
    return pd.DataFrame({"A": [1]})


@pytest.fixture
def dataframe_with_nan(test_settings: TestSettings) -> pd.DataFrame:
    """
    Provides a DataFrame containing NaN values for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        pd.DataFrame: DataFrame with NaN values.
    """
    np.random.seed(test_settings.SEED)
    data = np.random.rand(20)
    data[5:8] = np.nan
    return pd.DataFrame({"values": data})


@pytest.fixture
def multidimensional_array(test_settings: TestSettings) -> np.ndarray:
    """
    Provides a multidimensional numpy array for testing squeeze functionality.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        np.ndarray: Multidimensional array that can be squeezed.
    """
    np.random.seed(test_settings.SEED)
    return np.random.rand(1, 10, 1)


@pytest.fixture
def sample_binary_classification_data(test_settings: TestSettings) -> tuple:
    """
    Provides sample binary classification data for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        tuple: (y_true, probabilities, id2label)
    """
    np.random.seed(test_settings.SEED)
    y_true = np.random.randint(0, 2, 100)
    probabilities = np.random.rand(100, 2)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    id2label = {0: "Class_0", 1: "Class_1"}
    return y_true, probabilities, id2label


@pytest.fixture
def sample_multiclass_classification_data(test_settings: TestSettings) -> tuple:
    """
    Provides sample multiclass classification data for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.

    Returns:
        tuple: (y_true, probabilities, id2label)
    """
    np.random.seed(test_settings.SEED)
    y_true = np.random.randint(0, 3, 100)
    probabilities = np.random.rand(100, 3)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    id2label = {0: "Class_0", 1: "Class_1", 2: "Class_2"}
    return y_true, probabilities, id2label
