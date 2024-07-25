from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto-topic-gen",
    author="Sri Tikkireddy",
    author_email="sri.tikkireddy@databricks.com",
    description="Generate topics from sets of questions, predetermined topics or zero shot using various different techniques such as llms or bert based models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stikkireddy/auto-topic-gen",
    packages=find_packages(),
    install_requires=["pydantic>=2.8.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
