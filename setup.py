"""Setup for project package
example from https://github.com/pypa/sampleproject/blob/main/setup.py
"""

import pathlib
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(name='hdrml',
    version='0.1',
    description='Module for the project HDR with AI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://con.seri.co.uk:8090/pages/viewpage.' + \
        'action?spaceKey=AI&title=GN1%3A+Game+-+Standard+to+High+Dynamic+Range+Conversion',
    author='Juan Borrego-Carazo, Frederik Laboyrie, Cristian Szabo. GN1 Game. On Device AI. ',
    author_email='j.carazo@samsung.com',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=["hdrml"],
    # install_requires=['peppercorn'],
    python_requires='>=3.6, <4',
    zip_safe=False)
