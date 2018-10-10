import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ndmaze",
    version="0.0.1",
    author="Benjamin Bay",
    author_email="benjamin.bay@gmail.com",
    description="Package for generating and traversing mazes of 2 to n dimensions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BayBenj/ndmaze",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

