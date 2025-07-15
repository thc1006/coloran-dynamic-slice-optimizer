from setuptools import setup, find_packages

setup(
    name='5g-slice-optimizer',
    version='1.0.0',
    author='[Your Name]',
    author_email='[your.email@example.com]',
    description='GPU-accelerated dynamic resource allocation optimization for 5G network slicing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/[your-username]/5G-Slice-Optimizer',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Networking',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.2.0',
        'tensorflow-cpu',
        'pyarrow>=10.0.0',
        'joblib>=1.2.0',
        'openpyxl',
        'psutil>=5.9.0',
        'tqdm>=4.60.0',
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'pytest>=7.0.0',
    ],
    entry_points={
        'console_scripts': [
            'run-5g-slice-experiment=experiments.run_experiment:main',
        ],
    },
)
