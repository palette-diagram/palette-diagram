import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'scikit-learn',
    'numba',
    'matplotlib',
    'seaborn'
]


setuptools.setup(
    name="palette_diagram",
    version="1.1.2",
    author="Chihiro Noguchi and Tatsuro Kawamoto",
    author_email="chihiro3abc@gmail.com, kawamoto.tatsuro@gmail.com",
    description="Visualization tool for collective categorical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chnoguchi/pip_test.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.7',
)