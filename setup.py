"""Setup configuration for LLMFactory."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="LLMFactory",
    version="0.2.3",
    author="M Chimiste",
    author_email="",
    description="A unified factory pattern interface for multiple LLM inference providers with multimodal support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/M-Chimiste/LLMFactory",
    project_urls={
        "Bug Tracker": "https://github.com/M-Chimiste/LLMFactory/issues",
        "Source Code": "https://github.com/M-Chimiste/LLMFactory",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="llm ai machine-learning inference anthropic openai gemini ollama bedrock multimodal",
    include_package_data=True,
    zip_safe=False,
)