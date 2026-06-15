from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agromind-rag",
    version="1.0.0",
    description="Knowledge base for agricultural products and disease control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AgroMind Team",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.0",
        "langchain-chroma>=0.1.0",
        "langchain-ollama>=0.1.0",
        "chromadb>=0.5.0",
    ],
    python_requires=">=3.9",
)