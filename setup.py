from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

with open('LICENSE', 'r') as f:
    lic = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fbs',
    version='0.1.0',
    author='Zheng Zhao and Adrien Corenflos',
    author_email='zz@zabemon.com',
    keywords=['stochastic differential equations',
              'statistics',
              'MCMC'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    license=lic,
    packages=['fbs'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    python_requires='>=3.9',
    install_requires=requirements
)
