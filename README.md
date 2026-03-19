# Samsung - Deep Learning Image Quality Analysis (IQA)


>[!NOTE]
>
> This project is under active development in collaboration with Samsung

## The Goal
A specialized No-Reference Image Quality Assessment (IQA) model designed to automatically detect, classify, and quantify image degradation. This project focuses on identifying technical artifacts such as sensor faults and processing errors without requiring a "clean" reference image.

## Project Overview
In many real-world scenarios, a pristine reference image isn't available. This project implements a Convolutional Neural Network (CNN) approach to evaluate image quality "on the fly." Developed as part of an initiative to streamline quality control, this tool integrates directly into evaluation pipelines to ensure high-standard visual output.

## Key Features
* No-Reference Assessment: Evaluates quality based solely on the input image.

* Degradation Classification: Identifies specific types of noise, sensor artifacts, and processing distortions.

## Tech Stack
* Language: Python

* Architecture: Custom CNN 

* Data Handling: NumPy, Pandas

## Methodology
The model was trained to distinguish between high-quality captures and various levels of synthetic and real-world degradations:

1. Preprocessing: Normalization and artifact-specific augmentation.

2. Feature Extraction: Leveraging CNN layers to identify spatial anomalies.

3. Classification: Outputting a specific distortion category 