from pathlib import Path
from setuptools import setup, find_packages

requirements = [line.strip() for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]

setup(
    name="flow-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
