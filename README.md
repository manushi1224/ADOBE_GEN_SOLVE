﻿# ADOBE_GEN_SOLVE

# Shape Detection in CSV Data

This repository contains a Jupyter notebook that demonstrates various shape detection techniques, including detecting straight lines, circles, rectangles, rounded rectangles, polygons, and stars. The code processes point data from CSV files and visualizes the detected shapes.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Examples](#examples)

## Overview

The Jupyter notebook in this repository contains code to:

- Read point data from a CSV file.
- Detect shapes like straight lines, circles, rectangles, rounded rectangles, polygons, and stars.
- Plot and visualize the detected shapes with Matplotlib.

## Features

- **Straight Line Detection:** Identify and fit straight lines using linear regression.
- **Circle Detection:** Detect circles by fitting the points using a RANSAC-based method.
- **Rectangle and Rounded Rectangle Detection:** Identify rectangles and rounded rectangles based on contour analysis.
- **Polygon Detection:** Detect polygons with a specified number of sides.
- **Star Detection:** Identify star shapes using Convex Hull and angle analysis.

## Setup

### Prerequisites

Ensure you have the following packages installed:

- Python 3.x
- NumPy
- Matplotlib
- OpenCV (for rectangle detection)
- Scikit-learn (for linear regression and RANSAC)
- SciPy (for optimization)

You can install these dependencies using pip:

## Follow these steps to set up and activate the virtual environment and install the required packages.

### Create virtual environment

```shell
py -m venv .venv
```

### Activate the virtual environment

```shell
.venv\Scripts\activate
```

### Update pip

Make sure pip is up-to-date.

```shell
py -m pip install --upgrade pip
py -m pip --version
```

### Install packages

Install the required packages using the provided requirements file.

```shell
py -m pip install -r requirements.txt
```

That's it! You're now ready to start using the project.

## Usage

Open the Jupyter notebook:

```bash
jupyter notebook curvetopia.ipynb
```

### Structure of the Notebook

1. **Reading CSV Data:** The notebook begins with code to read and parse the CSV data into usable formats.
2. **Shape Detection Functions:** Includes functions to detect lines, circles, rectangles, rounded rectangles, polygons, and stars.
3. **Plotting Functions:** Functions to visualize the detected shapes are also included.
4. **Example Usage:** The notebook concludes with examples of detecting and plotting shapes from the provided CSV data.

## Examples

Here are a few examples of what the notebook can do:

- Detecting and Plotting a Circle:

[<img src="circle-reshape.png" width="400"/>](circle-reshape.png)

- Detecting a Star Shape:

[<img src="star-detection.png" width="400"/>](star-detection.png)

For more examples and detailed explanation, please refer to the Jupyter notebook.
