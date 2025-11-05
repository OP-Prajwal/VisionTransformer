# 🧠 VisionTransformer  
A Vision Transformer (ViT) model built **from scratch** using PyTorch — achieving **~97% validation accuracy** on the MNIST dataset.

---

## 🚀 Overview  
The **Vision Transformer (ViT)** applies transformer-based architectures (originally for NLP) to image classification tasks.  
Instead of using convolutional layers (like CNNs), ViT splits images into smaller patches, embeds them, and feeds them into a transformer encoder — learning global image representations efficiently.

This repository demonstrates a **minimal yet complete Vision Transformer**, implemented step-by-step for **educational and experimental** purposes.

---

## 🧩 Features
- ✅ Vision Transformer implemented from scratch (no high-level ViT libraries)  
- 📦 Works on **MNIST dataset**  
- 🏋️ Trained model weights included (`vision_transformer_mnist.pth`)  
- 📈 Achieves ~97% validation accuracy  
- 📚 Fully explained in a **Jupyter Notebook**  

---

## 📂 Repository Structure
```
VisionTransformer/
│
├── VisionTransformer.ipynb        # Main Jupyter Notebook implementation
├── vision_transformer_mnist.pth   # Trained model weights
└── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup
### Prerequisites
Make sure you have the following installed:
- Python 3.8 or above  
- PyTorch  
- torchvision  
- numpy, matplotlib, tqdm, jupyter

### Clone this repository
```bash
git clone https://github.com/OP-Prajwal/VisionTransformer.git
cd VisionTransformer
```

### Open the notebook
```bash
jupyter notebook VisionTransformer.ipynb
```

Then, execute the notebook cells step-by-step to:
1. Load the MNIST dataset  
2. Patch and embed images  
3. Train and evaluate the Vision Transformer model  
4. Save or load the trained weights

---

## 🧠 Model Architecture
### The Vision Transformer consists of:
1. **Patch Embedding Layer** – Splits 28×28 MNIST images into small patches and embeds them into vectors.  
2. **Positional Encoding** – Adds spatial information to patch embeddings.  
3. **Transformer Encoder Blocks** – Multi-head self-attention + feed-forward layers.  
4. **Classification Head** – Uses a special [CLS] token to predict the digit (0–9).

---

## 🧪 Example: Using Pre-trained Model
```python
import torch
from your_model_file import VisionTransformerModel  # Replace with actual model file name

model = VisionTransformerModel(...)
model.load_state_dict(torch.load("vision_transformer_mnist.pth"))
model.eval()

# Example inference
with torch.no_grad():
    output = model(img)  # img: (1, 1, 28, 28)
    predicted_label = output.argmax(dim=1)
print("Predicted Digit:", predicted_label.item())
```

---

## 📈 Results
| Metric | Value |
|--------:|-------|
| Dataset | MNIST |
| Accuracy | ~97% |
| Model | Vision Transformer |
| Framework | PyTorch |

---

## 💡 Future Enhancements
- Support for **CIFAR-10 / CIFAR-100** datasets  
- Add **data augmentation** and **regularization**  
- Visualize **attention maps**  
- Convert notebook to standalone Python script  
- Integrate with a simple web app (e.g., Streamlit demo)

---

## 📚 References
- [📄 An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## 🧑‍💻 Author
**Prajwal (OP-Prajwal)**  
🔗 [GitHub Profile](https://github.com/OP-Prajwal)

---

## 📜 License
This project is open for **educational and research purposes**.  
If you modify or use it in your own projects, please credit this repository.

---

⭐ **If you like this project, don’t forget to give it a star on GitHub!**
