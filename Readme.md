### Abstract
Medical image datasets, such as COVID-19 chest CT scans, are difficult to share across hospitals because of privacy, legal, and ethical constraints. Federated Learning (FL) enables collaborative training without moving raw data; however, traditional FL methods like FedAvg require exchanging full model weights, which is computationally heavy and potentially vulnerable to model inversion attacks.

This project proposes a *Lightweight and Privacy-Preserving Federated Learning Framework* for COVID-19 CT image classification. Instead of sharing model weights, each client extracts a compact feature embedding and applies two security mechanisms:  
1. **Top-K Feature Sparsification** – reduces transmitted dimensions by keeping only the most informative 10% of features.  
2. **Feature-Level Differential Privacy (Gaussian Noise Mechanism)** – adds calibrated noise to embeddings to prevent reconstruction of original CT images.

The server aggregates these noisy sparse prototype vectors and distributes global prototypes back to clients, enabling lightweight collaborative learning. Experiments show that the framework significantly reduces communication overhead while maintaining good accuracy and strong privacy guarantees.

### 1. Introduction
COVID-19 rapidly increased the demand for automated medical diagnosis systems capable of analyzing chest CT images. However, building accurate AI models requires large and diverse datasets, which individual hospitals often lack. Sharing patient images across institutions is restricted due to privacy regulations (HIPAA, GDPR), ethical concerns, and administrative barriers.

Federated Learning (FL) provides a solution by allowing multiple clients (hospitals) to train a shared model without sending raw data. Traditional FL techniques like FedAvg share entire model parameters every round, which has three major problems:

1. **High Communication Cost** – deep models contain millions of parameters.  
2. **Architecture Dependency** – all clients must use identical neural networks.  
3. **Privacy Vulnerability** – shared weights or embeddings can be reconstructed through model inversion attacks.

To address these limitations, this project implements a *Lightweight Federated Learning System* where:

- Clients share **only feature vectors**, not model weights.  
- Features undergo **Top-K sparsification** to reduce size.  
- Differential Privacy noise protects against reconstruction attacks.  

This approach reduces communication, increases privacy, and allows heterogeneous models on different clients.

## 2. Literature Review

### 2.1 COVID-19 Detection Using Medical Imaging
Deep learning has been widely applied to COVID-19 diagnosis using chest X-rays (CXR) and CT images. Early approaches used transfer learning on popular CNN architectures such as:

- VGG16 / VGG19  
- ResNet18 / ResNet50  
- DenseNet121  
- MobileNet and EfficientNet  

These models achieve high accuracy but rely heavily on large labeled datasets. Public datasets such as COVID-QU-Ex, COVIDx-CT, and CC-CCII improved research accessibility but remain insufficient for clinical-scale systems.

### 2.2 Classical Federated Learning (FedAvg)
Federated Learning (FL) was introduced as a privacy-friendly paradigm where clients train models locally and only exchange model parameters. The most widely used method is:

**Federated Averaging (FedAvg)**  
- Clients download a global model  
- Train locally  
- Upload updated weights  
- Server averages the weights  

However, FedAvg has limitations:

1. Requires identical model architectures across clients  
2. Transmits millions of parameters per round  
3. Vulnerable to gradient leakage and model inversion attacks

These factors make FedAvg unsuitable for hospitals with limited bandwidth or heterogeneous hardware.

### 2.3 Lightweight Federated Learning
To address communication overhead, recent research explored:

- **Partial model sharing** (sending only some layers)  
- **Compressed gradients**  
- **Knowledge distillation between clients**  
- **Prototype-based FL** (clients send feature embeddings instead of weights)  

Prototype-based FL significantly reduces communication cost and allows heterogeneous models, making it ideal for real-world medical systems.

### 2.4 Federated Learning for COVID-19
Recent works apply FL to COVID-19 detection using CT or X-ray images. However:

- Most use FedAvg (heavy communication).  
- Few works address privacy attacks like model inversion.  
- Majority assume identical model architecture across hospitals.  

The paper “Lightweight Federated Learning for Detecting COVID-19 in Chest CT Images” introduced a feature-vector–based communication system to reduce overhead, but **it lacked privacy protection**, making it vulnerable to reconstruction attacks.

### 2.5 Privacy Attacks on FL
Modern papers show that attackers can reverse-engineer:

- Images from gradients  
- Images from embeddings  
- Images from intermediate activations  

This exposes a major risk when sharing feature vectors.

### 2.6 Differential Privacy in FL
Differential Privacy (DP) is the most effective method to defend against reconstruction attacks. It adds controlled noise to data representations and provides a mathematical privacy guarantee.

However:

- DP-SGD is computationally heavy  
- It slows down training  
- It requires modifying the optimizer

Thus, a **lightweight alternative** is desirable.

### 2.7 Gap in Existing Research
Most FL works for COVID-19:

- Do not support heterogeneous models  
- Do not protect against reconstruction attacks  
- Do not reduce communication cost effectively  

**Your project fills this gap by combining prototype-based FL + sparsification + feature-level differential privacy.**

## 3. Methodology

This project proposes a **Lightweight, Privacy-Preserving Federated Learning Framework** for COVID-19 CT image classification. Unlike traditional Federated Learning (FedAvg), which transmits full model parameters, our system transmits only compact and privacy-protected **feature vectors**. The methodology consists of four major components:

1. Local Model Training on the Client  
2. Feature Extraction  
3. Privacy Protection (Sparsification + Differential Privacy)  
4. Secure Aggregation on the Server  


---

### 3.1 System Architecture Overview

The system contains:

- **N Clients (Hospitals)**  
  Each client holds its own private CT images and trains its own model locally.

- **Central Server**  
  Aggregates privacy-protected feature vectors and produces global feature prototypes.

---

### 3.2 Local Client Model

Each client uses a **Lightweight Convolutional Neural Network** designed with:

- Feature extraction block  
- Classification head  

The model is divided into two conceptual parts:

1. **Feature Extractor (Local, private)**  
   Generates feature embeddings of images.

2. **Classifier (Local)**  
   Used only for client-side prediction and validation.

---

### 3.3 Local Training Phase

Each client performs standard supervised training:

1. Load local CT dataset  
2. Forward pass through model  
3. Compute classification loss  
4. Backpropagate gradients  
5. Update the model locally  

The server does **not** receive any gradients or model weights.

---

### 3.4 Feature Extraction

After local training, each client extracts embeddings:

\[
f_i = ExtractFeatures(x_i)
\]

Where:

- \( x_i \) = CT image  
- \( f_i \) = feature vector (e.g., 256-dimensional)

These feature vectors serve as the “knowledge” each client contributes to the federated system.

---

### 3.5 Feature Sparsification (Top-K Selection)

To reduce communication cost and increase privacy, we perform **Top-K sparsification**:

- Only the largest \( K \% \) of feature values are retained  
- Rest are set to zero  

Example:

```
keep top 10% values → 90% removed
```

Benefits:

- Reduces feature dimension  
- Removes sensitive components  
- Makes reconstruction attacks significantly harder  

---

### 3.6 Feature-Level Differential Privacy (Gaussian Mechanism)

This is the core privacy innovation in your project.

Each feature vector after sparsification gets **Gaussian noise**:

\[
f' = f + \mathcal{N}(0, \sigma^2)
\]

Where:

- \( f' \) = DP-protected feature vector  
- \(\sigma = \frac{||f||_2}{\epsilon}\)  
- \( \epsilon \) = privacy budget  

This ensures **(ε, δ)-Differential Privacy**.

Advantages:

- Lightweight  
- Does not slow down model training  
- Strong protection against model inversion  

This is superior to the original COVID-FL paper, which had **no real privacy protection**.

---

### 3.7 Transmission to Server

Clients send only:

- **Sparsified**
- **DP-protected**
- **Feature vectors**

Meaning no raw images, no gradients, no model weights are transmitted.

This drastically reduces communication to **only a few kilobytes per round**.

---

### 3.8 Secure Server Aggregation

The server computes a simple mean:

\[
F_{global} = \frac{1}{N} \sum_{i=1}^N f'_i
\]

This aggregated feature prototype represents global knowledge.

---

### 3.9 Global Feature Distribution

The server sends back:

- Only the aggregated prototype  
- No client-specific data  
- No sensitive information  

Each client then uses the global prototype to improve local training in the next round.

---

### 3.10 Summary of Methodology Flow

1. **Client Training**  
   Local model updates using private CT images.

2. **Feature Extraction**  
   Convert CT images into embeddings.

3. **Sparsification**  
   Keep only top 10% important features.

4. **Differential Privacy**  
   Add Gaussian noise to protect identity.

5. **Upload**  
   Send protected features to server.

6. **Aggregation**  
   Server averages all client feature vectors.

7. **Broadcast**  
   Server sends global prototype back.

This architecture is **lightweight, secure, and suitable for real hospitals.**

