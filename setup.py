from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

long_description = """
Bibliometric tools to analyze data downloaded from sources such
as Reuters' Web of Science (right now the only source supported).
Used by the Marion web platform for bibliometrics analysis.
"""

setup(
    name='marion-biblio',
    url='https://github.com/vputz/marion-biblio',
    version='1.0.0',
    description='simple tools for bibliometrics analysis',
    long_description=long_description,
    author='Victor Putz',
    author_email='vputz@nyx.net',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        ],

    keywords='bibliometrics',

    packages=['marion_biblio'],

    install_requires=[
        'numpy==1.9.2',
        'tables==3.2.1',
        'pytest-bdd==2.14.4',
        'scipy==0.16.0',
        'pandas==0.16.2'
    ],

    extras_require={},

    package_data={},

    data_files=[],

    entry_points={
            },
)
