# CAD 3D Model Classification and Generation using Graph Neural Networks

## Authors and Contributions
This repository is an adaptation and extension of the original work by Lorenzo Mandelli and Stefano Berretti at Università degli Studi di Firenze, focusing on 3D CAD model classification using Graph Neural Networks (GNNs). 

### **Contributions by SeungJe Woo**:
- **Developed a CAD file generation module**: Designed and implemented a method to create CAD files using a **Variable Graph Autoencoder (VGAE)**.
- **Integrated generative modeling with classification tasks**: Extended the repository's capabilities to generate and classify 3D CAD models, ensuring compatibility with STEP file formats.
- **Enhanced usability and modularity**: Improved the structure and usability of the repository to support additional workflows, including generative model training.

The original repository focused on the classification of 3D CAD models in their native STEP format. My contributions add a generative modeling component using VGAEs, enabling the creation of synthetic CAD models for downstream tasks such as training, validation, and benchmarking.

---

## Abstract
This repository now supports both the **classification** and **generation** of 3D CAD models. Classification leverages Graph Neural Networks (GNNs) to analyze and retrieve models in the STEP format, while generation uses a Variable Graph Autoencoder (VGAE) to create synthetic CAD models, enriching datasets for more robust analysis. By combining classification and generation, this repository provides an integrated framework for CAD-based machine learning tasks.

### Original Abstract (Mandelli & Berretti):
*In this paper, we introduce a new approach for retrieval and classification of 3D models that directly performs in the CAD format without any format conversion to other representations like point clouds or meshes, thus avoiding any loss of information. Among the various CAD formats, we consider the widely used STEP extension, which represents a standard for product manufacturing information. This particular format represents a 3D model as a set of primitive elements such as surfaces and vertices linked together. In our approach, we exploit the linked structure of STEP files to create a graph in which the nodes are the primitive elements and the arcs are the connections between them. We then use Graph Neural Networks (GNNs) to solve the problem of model classification.*  

Building on this foundation, I extended the repository to support **generative modeling with VGAEs**.

---

## Installation

Follow the original installation steps provided in the repository and add the following for the generative model:

1. Install additional requirements for VGAE:
   ```bash
   conda install pytorch-lightning
   conda install -c conda-forge networkx
   ```

2. Prepare datasets for generative modeling:
   - Ensure that datasets include labeled graphs representing CAD components (STEP format is supported).

---

## Usage

### Graph-Based Classification
Refer to the original instructions for graph-based classification using `step_2_graph.py` and `train_GCN.py`.

### Generative Modeling with VGAE
1. **Train the VGAE model**:
   ```bash
   python train_vgae.py --dataset_path <path_to_dataset> --output_path <output_directory> --epochs <num_epochs>
   ```

2. **Generate new CAD models**:
   ```bash
   python generate_cad.py --model_path <trained_model_path> --num_samples <number_of_samples>
   ```

3. **Visualize generated models**:
   Use the script `visualize_cad.py` to render and analyze generated CAD models.

### Example Workflow
1. Convert STEP files to graphs using `step_2_graph.py`.
2. Train a VGAE model using `train_vgae.py`.
3. Generate synthetic CAD models and classify them using `train_GCN.py` for evaluation.

---

## Repository Requirements
This repository builds on the original implementation, which uses PyTorch and CUDA Toolkit version 11.3 for GPU computations. The generative model component additionally requires PyTorch Lightning and NetworkX.

---

## Acknowledgments
This repository is adapted from the original work by Lorenzo Mandelli and Stefano Berretti. Please cite their work if you use this repository in your research:

```
@misc{https://doi.org/10.48550/arxiv.2210.16815,
  doi = {10.48550/ARXIV.2210.16815},
  url = {https://arxiv.org/abs/2210.16815},
  author = {Mandelli, L. and Berretti, S.},
  title = {CAD 3D Model classification by Graph Neural Networks: A new approach based on STEP format},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

Additionally, please acknowledge my contributions if you use the generative modeling component:

```
@misc{woo2024cad,
  author = {SeungJe Woo},
  title = {Generative Modeling of CAD Files using Variable Graph Autoencoders},
  year = {2024},
  note = {Extension of original work by Mandelli and Berretti},
}
```
