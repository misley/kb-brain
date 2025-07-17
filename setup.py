from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kb-brain",
    version="1.1.0",
    author="NPS Data Science Team",
    author_email="noreply@nps.gov",
    description="Intelligent Knowledge Base Brain with Hierarchical SME Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/misley/kb-brain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "cupy-cuda12x>=12.0.0",
            "cuml-cu12>=24.0.0",
        ],
        "continue": [
            "aiohttp>=3.8.0",
            "requests>=2.25.0",
        ],
        "intelligence": [
            "asyncio-mqtt>=0.13.0",
            "aiofiles>=23.0.0",
        ],
        "sme": [
            "networkx>=3.0",
            "matplotlib>=3.6.0",
        ],
        "performance": [
            "intel-extension-for-scikit-learn>=2024.0.0",
            "numba>=0.58.0",
            "scipy>=1.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
        ],
        "all": [
            "cupy-cuda12x>=12.0.0",
            "cuml-cu12>=24.0.0",
            "aiohttp>=3.8.0",
            "requests>=2.25.0",
            "asyncio-mqtt>=0.13.0",
            "aiofiles>=23.0.0",
            "networkx>=3.0",
            "matplotlib>=3.6.0",
            "intel-extension-for-scikit-learn>=2024.0.0",
            "numba>=0.58.0",
            "scipy>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kb-brain=kb_brain.cli.main:main",
            "kb-brain-mcp=kb_brain.mcp.server:main",
            "kb-brain-monitor=kb_brain.cli.monitor:main",
            "kb-brain-continue=kb_brain.cli.continue_cli:continue_cli",
            "kb-brain-intelligence=kb_brain.intelligence.intelligence_system:main",
            "kb-brain-sme=kb_brain.intelligence.sme_agent_system:main",
            "kb-brain-consolidate=kb_brain.intelligence.kb_consolidation_system:main",
            "kb-brain-optimize=kb_brain.performance.cpu_optimizer:main",
            "kb-brain-benchmark=kb_brain.performance.performance_integration:main",
        ],
    },
    include_package_data=True,
    package_data={
        "kb_brain": [
            "config/*.yaml",
            "config/*.json",
            "scripts/*.sh",
        ],
    },
)
