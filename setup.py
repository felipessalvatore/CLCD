from setuptools import setup, find_packages

setup(
    name='CLCD',
    version='0.1',
    description='Cross-Lingual Contradiction Detection', # noqa
    author='Felipe Salvatore',
    packages=find_packages(),
    test_suite="tests"
)
