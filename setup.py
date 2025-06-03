#!/usr/bin/env python
"""Setup script for PyTorch Memory Profiler."""

import os
from setuptools import setup, find_packages

# Read the README file for the long description
def read_long_description():
    """Read the README file for the long description."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# Core dependencies
install_requires = [
    'torch>=1.12.0',  # Required for FakeTensorMode and TorchDispatchMode
]

# Optional dependencies
extras_require = {
    'rich': [
        'rich>=10.0.0',  # For enhanced log formatting and visual output
    ],
    'megatron': [
        'megatron-core',  # For tracing memory in models using Megatron's pipeline parallelism
    ],
    'te': [
        'transformer-engine',  # For tracing memory in models using Transformer Engine layers
    ],
    'dev': [
        'pytest>=6.0',
        'pytest-cov',
        'black',
        'flake8',
        'isort',
        'mypy',
    ],
    'all': [
        'rich>=10.0.0',
        # Note: megatron-core and transformer-engine are not included in 'all'
        # because they have specific installation requirements and dependencies
    ],
}

setup(
    name='pytorch-memory-profiler',
    version='0.1.0',
    author='PyTorch Memory Profiler Contributors',
    author_email='',
    description='A utility library for estimating and analyzing the memory footprint of PyTorch models',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/PyTorch-Memory-Profiler',  # Update with actual URL
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/PyTorch-Memory-Profiler/issues',
        'Source': 'https://github.com/yourusername/PyTorch-Memory-Profiler',
    },
    packages=find_packages(exclude=['examples', 'tests', 'docs', 'tools']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  # Update if different license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='pytorch memory profiling deep-learning neural-networks gpu-memory',
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            # Add any command-line scripts here if needed
            # 'pytorch-memory-profiler=memory_profiler.cli:main',
        ],
    },
) 