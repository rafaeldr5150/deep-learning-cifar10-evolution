# deep-learning-cifar10-evolution

🚀 From Scratch to Transfer Learning: A Progressive Deep Learning Journey on CIFAR-10

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## 🎯 About The Project

This comprehensive portfolio demonstrates a structured learning progression through modern deep learning techniques, starting from basic convolutional neural networks and advancing to state-of-the-art transfer learning architectures. All projects use the CIFAR-10 dataset to provide consistent benchmarking across different approaches.

**Why this project exists:**
- To showcase a complete deep learning learning path
- To compare and contrast different neural network architectures
- To demonstrate practical implementation of theoretical concepts
- To provide reproducible examples for educational purposes

**Technical Environment:**
- 🔥 **Google Colab Pro+** for accelerated GPU training
- 💾 Extensive model persistence and backup strategies
- ⚡ Optimized for high-performance computing environments

**Key Features:**
- 🔍 **Progressive Difficulty**: Starts simple, advances to complex
- 📊 **Consistent Benchmarking**: Same dataset across all experiments
- 🎯 **Practical Focus**: Real-world implementation details
- 📈 **Visual Analytics**: Comprehensive results visualization

## 🛠️ Built With

### Core Technologies
- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
- ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
- ![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

### Data Science Stack
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
- ![Scikit-learn](https://img.shields.io/badge/Scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or Google Colab account

### 💰 Computing Resources
This project was developed using **Google Colab Pro+** to leverage:
- A100/T4 GPU acceleration for faster training
- Extended session times for complex model architectures
- Enhanced memory for large-scale experiments

## 🔧 Technical Environment

### Cloud Computing Platform
- **Google Colab Pro+** with premium GPU access
- A100/T4 Tensor Core GPUs for accelerated training
- High-RAM runtime for complex model architectures

### Why This Matters:
- Enabled training of state-of-the-art architectures
- Reduced training time from days to hours
- Supported extensive hyperparameter tuning
- Facilitated model persistence and versioning

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rafaeldr5150/deep-learning-cifar10-evolution.git
   cd deep-learning-cifar10-evolution
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv dl-env
   source dl-env/bin/activate  # On Windows: dl-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt:**
   ```
   tensorflow>=2.12.0
   keras>=2.12.0
   numpy>=1.21.0
   matplotlib>=3.5.0
   seaborn>=0.11.0
   scikit-learn>=1.0.0
   pandas>=1.3.0
   jupyter>=1.0.0
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### Alternative: Google Colab Setup

1. **Upload each notebook** to your Google Drive
2. **Open with Google Colab**
3. **Ensure GPU runtime**: Runtime → Change runtime type → GPU
4. **Run cells sequentially**

## 📁 Project Structure

```
deep-learning-cifar10-evolution/
│
├── 📓 CNN_Model.ipynb
├── 📓 MobileNetV2.ipynb
├── 📓 densenet_cifar_version1.ipynb
├── 📓 FinalefficientB0.ipynb
├── 📓 efficienentnetv2_bo.ipynb
│
├── 📊 Presentation.pdf
│
├── 📄 LICENSE
└── 📖 README.md
```

## 📊 Results

### Performance Summary

| Architecture | Test Accuracy | Training Time | Parameters | Key Strength |
|--------------|---------------|---------------|------------|--------------|
| Custom CNN | 79% | Fast | ~225K | Fundamentals |
| MobileNetV2 | 91% | Medium | ~2.2M | Efficiency |
| DenseNet121 | 94% | Long | ~7M | Feature Reuse |
| EfficientNetB0 | 97% | Medium | ~4M | Optimization |
| EfficientNetV2B0 | 97% | Medium | ~6M | State-of-the-Art |

### Key Findings

1. **Data Augmentation Impact**: +13.6% improvement in custom CNN
2. **Transfer Learning Power**: 91% vs 79% starting point
3. **Architecture Matters**: Clear accuracy progression across models
4. **Efficiency Trade-offs**: Different balance of accuracy vs speed

## 💻 Usage

Each notebook is self-contained and demonstrates specific concepts:

### Notebook 1: CNN Fundamentals
- **File**: `CNN_Model.ipynb`
- **Focus**: Building neural networks from scratch
- **Key concepts**: Data augmentation, regularization, overfitting

### Notebook 2: Transfer Learning Introduction  
- **File**: `MobileNetV2.ipynb`
- **Focus**: Leveraging pre-trained models
- **Key concepts**: Fine-tuning, feature extraction

### Notebook 3: Advanced Architectures
- **File**: `densenet_cifar_version1.ipynb`
- **Focus**: Complex network design
- **Key concepts**: Dense connections, selective freezing

### Notebook 4-5: Modern Approaches
- **Files**: `FinalefficientB0.ipynb`, `efficienentnetv2_bo.ipynb`
- **Focus**: State-of-the-art efficiency
- **Key concepts**: Compound scaling, optimized training

### Running a Specific Notebook
```bash
jupyter notebook CNN_Model.ipynb
```

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

### How to Contribute

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Additional architecture implementations
- Performance optimizations
- Enhanced visualizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.

## 👨‍💻 Authors

**Rafael Rocha** - *Initial work* - https://github.com/rafaeldr5150

## 🙏 Acknowledgements

### Resources
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - University of Toronto
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Examples](https://keras.io/examples/)

### Inspiration
- [freeCodeCamp](https://www.freecodecamp.org/) - For the excellent README guide
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Research papers on EfficientNet and DenseNet architectures

### Tools
- Google Colab for GPU acceleration
- Matplotlib and Seaborn for visualization
- scikit-learn for evaluation metrics

---

<div align="center">

### ⭐ Don't forget to star this repository if you found it helpful!

</div>
