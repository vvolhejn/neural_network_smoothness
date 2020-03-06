from setuptools import setup, find_packages

setup(
    name="smooth",
    version="0.1",
    description="Exploration of double descent",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "smooth=smooth.main:main",
        ]
    }, install_requires=['tensorflow', 'numpy', 'pandas', 'scipy', 'sacred', 'tqdm',
                         'scikit-learn', 'GPy', 'PyYAML']
)