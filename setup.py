"""Setup file for the Plexure API Search package."""

from setuptools import setup, find_packages

setup(
    name="plexure-api-search",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "httpx",
        "pinecone-client",
        "rich",
        "sentence-transformers",
        "pyyaml",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "plexure-api-search=plexure_api_search.__main__:main",
        ],
    },
    author="Pimentel",
    author_email="pimentel@example.com",
    description="A powerful semantic search tool for API contracts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pimentel/plexure-api-search",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 