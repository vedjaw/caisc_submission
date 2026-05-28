# Data Description

> All datasets are downloaded automatically by the code on first run. PathMNIST
> can be slow to download from the medmnist servers — just be patient.

## Datasets Used

### 1. CIFAR-10
- **Source**: torchvision.datasets.CIFAR10 (auto-downloads)
- **Description**: 60,000 32×32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Train/Test Split**: 50,000 training / 10,000 test
- **Preprocessing**:
  - Random horizontal flip + random crop with 4px padding (training only)
  - Resize to 224×224 for ResNet input
  - Normalize: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
- **Forget Set**: All training samples of the specified class (5,000 samples per class)
- **Retain Set**: Training set minus forget set (45,000 samples)
- **Note**: This is our primary benchmark. Most of the hyperparameter tuning was done here.

### 2. PathMNIST (medical imaging)
- **Source**: medmnist.PathMNIST (MedMNIST v2 package)
- **Description**: 107,180 images of colorectal cancer histology patches in 9 tissue types
- **Original Resolution**: 28×28 (resized to 224×224 for ResNet)
- **Train/Val/Test Split**: 89,996 / 10,004 / 7,180
- **Classes**: 9 tissue types (adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium)
- **Preprocessing**: Resize to 224×224, convert to RGB (as_rgb=True), ImageNet normalization
- **Quirk**: MedMNIST uses `.labels` instead of `.targets`, which broke our CIFAR-style
  data splitting code. We wrote a small shim (`_attach_medmnist_targets`) to fix this.
- **Forget Set**: Same class-level scheme as CIFAR-10

### 3. MNIST
- **Source**: torchvision.datasets.MNIST
- **Description**: 70,000 28×28 grayscale images of handwritten digits (0–9)
- **Train/Test Split**: 60,000 training / 10,000 test
- **Preprocessing**:
  - Convert grayscale → 3-channel RGB (repeat across channels)
  - Resize to 224×224 (for ResNet input)
  - Normalize: mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081]
- **Forget Set Construction**: Same as CIFAR-10 (class-level or sample-level)

### 4. SVHN
- **Source**: torchvision.datasets.SVHN
- **Description**: ~600,000 32×32 color images of house numbers from Google Street View, 10 digit classes (0–9)
- **Train/Test Split**: 73,257 training / 26,032 test (core set)
- **Preprocessing**:
  - Resize to 224×224 (for ResNet input)
  - Normalize: mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]
- **Forget Set Construction**: Same as CIFAR-10 (class-level or sample-level)

### 5. STL10
- **Source**: torchvision.datasets.STL10
- **Description**: 13,000 96×96 color images in 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- **Train/Test Split**: 5,000 training / 8,000 test
- **Preprocessing**:
  - Resize to 224×224 (for ResNet input)
  - Normalize: mean=[0.4467, 0.4398, 0.4066], std=[0.2242, 0.2215, 0.2239]
- **Forget Set Construction**: Same as CIFAR-10 (class-level or sample-level)
- **Note**: Smaller training set means each class has only ~500 training samples

## Reduced Setup / Constraints

- All experiments use ResNet-18 (11.7M params) because that's what we could run in
  reasonable time on a single GPU. Scaling to ResNet-50 or ViT is future work.
- Images: CIFAR-10/SVHN are 32×32 (resized to 224), MNIST is 28×28 → 224,
  STL10 is 96×96 → 224, PathMNIST is 28×28 → 224. Everything gets resized
  to 224 for the ResNet input layer.
- Grayscale datasets (MNIST) are converted to 3-channel RGB by repeating across channels.
- No data augmentation during unlearning, only during base model training.
- MNIST and PathMNIST were tested across all classes. CIFAR-10, SVHN, and STL10 were
  only tested on class 0 due to time constraints — ideally we'd test all 10 classes
  on each, but each run takes ~3 minutes and we had 5 datasets to cover.
