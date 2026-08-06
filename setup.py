# setup.py
from setuptools import setup, find_packages

setup(
    name='astra',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={'astra': ['hmm_databases.json']},
    install_requires=[
        'pyhmmer',
        'pandas',
        'tqdm',
        'requests',
        'platformdirs',
    ],
        entry_points={
            'console_scripts': ['astra=astra.main:main'],
        },
)
