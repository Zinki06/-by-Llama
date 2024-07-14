!pip install transformers==4.38.0 torch==2.3.0 torchvision==0.18.0 pillow==10.0.0 huggingface_hub

# 필요한 라이브러리 설치
!pip install transformers==4.38.0 torch==2.3.0 torchvision==0.18.0 pillow==10.0.0 huggingface_hub

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from google.colab import drive
from huggingface_hub import login
from torch.cuda.amp import autocast, GradScaler


hf_token = "hugging_face"

# Hugging Face 로그인
login(token=hf_token, add_to_git_credential=True)

# Google Drive 마운트
drive.mount('/content/drive')

# 이미지 데이터셋 클래스 정의
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_dir):
                images.append((os.path.join(class_dir, img_name), self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Google Drive 경로 설정
data_dir = '/content/drive/MyDrive/DEV/school_place/dataset/'  # 데이터가 저장된 Google Drive 경로
model_save_dir = '/content/drive/MyDrive/DEV/school_place/model/'  # 모델을 저장할 Google Drive 경로

# 데이터셋 및 데이터로더 생성
dataset = ImageFolderDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # 배치 크기를 16으로 줄임

# CNN 모델 정의 (ResNet18 사용)
cnn_model = models.resnet18(weights="IMAGENET1K_V1")
cnn_model.fc = nn.Identity()  # 마지막 fully connected 층 제거

# Llama3-8B-Instruct 모델 및 토크나이저 로드
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

# Add pad token if not already added
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_model.resize_token_embeddings(len(tokenizer))

# 멀티모달 분류기 정의
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, cnn_out_dim=512, llama_out_dim=4096):
        super().__init__()
        self.cnn = cnn_model
        self.llama = llama_model
        self.classifier = nn.Linear(cnn_out_dim + llama_out_dim, num_classes)

    def forward(self, images, text_inputs):
        cnn_features = self.cnn(images)
        llama_outputs = self.llama(**text_inputs)
        llama_features = llama_outputs.logits[:, 0, :]  # logits 사용
        combined_features = torch.cat((cnn_features, llama_features), dim=1)
        return self.classifier(combined_features)

# 모델 초기화
model = MultimodalClassifier(num_classes=len(dataset.classes))

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Mixed Precision Training을 위한 GradScaler
scaler = GradScaler() if torch.cuda.is_available() else None

# 학습 루프
num_epochs = 5
accumulation_steps = 2  # Gradient Accumulation을 위한 스텝 수
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for i, (batch_images, batch_labels) in enumerate(dataloader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # 이미지에 대한 텍스트 설명 생성 (실제로는 이미지 캡셔닝 모델을 사용해야 함)
        image_descriptions = [f"This is an image of {dataset.classes[label]}" for label in batch_labels]
        
        # 텍스트 토큰화
        text_inputs = tokenizer(image_descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Mixed Precision Training 적용
        if torch.cuda.is_available():
            with autocast():
                outputs = model(batch_images, text_inputs)
                loss = criterion(outputs, batch_labels)
                loss = loss / accumulation_steps  # Gradient Accumulation을 위해 손실을 나눔

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(batch_images, text_inputs)
            loss = criterion(outputs, batch_labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

print("Training completed!")


os.makedirs(model_save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_save_dir, 'multimodal_model.pth'))
tokenizer.save_pretrained(model_save_dir)

print(f"Model saved to Google Drive: {model_save_dir}")
