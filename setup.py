from setuptools import setup, find_packages

setup(
    name="pridictior_snow5",
    version="1.0.0",
    description="Bitcoin 25-Minute Real-Time Price Predictor with Market Simulator and Snowflake Notebook",
    author="fgbvdcfgbfdefgb",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "matplotlib>=3.7.0",
    ],
)
