# Quicktune Tool for Image Segmentation

Machine learning practitioners today face the dual challenge of selecting the most appropriate model and optimizing its fine-tuning process for specific tasks. The **[Quicktune Tool (QTT)](https://github.com/automl/quicktunetool)** optimizes the model selection, fine-tuning methodology, and hyperparameter optimization (HPO) process by leveraging meta-learning to analyze performance curves of pipelines across a wide variety of classification datasets.

While the methodology has shown effectiveness for **image classification**, this project aims to critically examine the performance of the algorithm and explore its potential in **image segmentation tasks**.

## Results & Visual Analysis

### <Insert image 1 here>
*Placeholder for image 1 analysis, to be filled in later.*

### <Insert image 2 here>
*Placeholder for image 2 analysis, to be filled in later.*

[Click here to read the full analysis report.](#)

## Testing Quicktune Tool for Image Segmentation

To test QTT for image segmentation, follow the steps below:

### Requirements:
- Clone the repository:

```bash
git clone https://github.com/nobara-netizen/QTT_SEG.git
cd QTT_SEG
```

- Create and activate the conda environment:

```bash
conda create -n qtt_seg python=3.11
conda activate qtt_seg
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Set up third-party dependencies:

```bash
mkdir third_party && cd third_party
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd ../..
```

- Run the main script:

```bash
python main.py
```

---

We look forward to your feedback and contributions as we explore the potential of Quicktune Tool for image segmentation tasks!
