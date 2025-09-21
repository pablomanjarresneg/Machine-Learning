from setuptools import setup, find_packages

setup(
    name="astro-ml-tools",
    version="0.1.0",
    description="Machine learning tools for astronomical data analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "astropy>=5.0.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "deep_learning": [
            "tensorflow>=2.8.0",
            "keras>=2.8.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)