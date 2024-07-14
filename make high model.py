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

# Hugging Face 토큰 설정 (토큰은 Hugging Face 계정에서 생성하여 사용)
hf_token = "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"

# Hugging Face에 로그인
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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN 모델 정의 (ResNet18 사용)
cnn_model = models.resnet18(weights="IMAGENET1K_V1")
cnn_model.fc = nn.Identity()  # 마지막 fully connected 층 제거

# Llama3-8B-Instruct 모델 및 토크나이저 로드
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

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
        llama_features = llama_outputs.last_hidden_state[:, 0, :]  # CLS token
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

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # 이미지에 대한 텍스트 설명 생성 (실제로는 이미지 캡셔닝 모델을 사용해야 함)
        image_descriptions = [f"This is an image of {dataset.classes[label]}" for label in batch_labels]
        
        # 텍스트 토큰화
        text_inputs = tokenizer(image_descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # 모델 실행
        outputs = model(batch_images, text_inputs)
        
        # 손실 계산 및 역전파
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

print("Training completed!")

# 모델을 Google Drive에 저장
os.makedirs(model_save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_save_dir, 'multimodal_model.pth'))
tokenizer.save_pretrained(model_save_dir)

print(f"Model saved to Google Drive: {model_save_dir}")
