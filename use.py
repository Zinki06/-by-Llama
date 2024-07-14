import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 커스텀 분류기 정의 (ResNet18 기반)
class CustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# 클래스 이름과 설명 정의
class_info = {
    "last_alley": "마지막 복도.",
    "middle_alley": "중간 복도.",
    "first_alley": "첫 번째 복도."
}
class_names = list(class_info.keys())
num_classes = len(class_names)

# 모델 로드
model_save_dir = '/content/drive/MyDrive/DEV/school_place/model/'

# 사용자에게 모델 파일 이름 입력 받기
while True:
    model_filename = input("Enter the name of the model file (e.g., custom_model.pth): ").strip()
    model_path = os.path.join(model_save_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        continue
    
    try:
        model = CustomClassifier(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        break
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please try again with a different file name.")

# GPU 사용 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 예측 함수
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return class_names[predicted.item()], probabilities[0]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# 예측 결과 해석 함수
def interpret_prediction(prediction, probabilities):
    confidence = probabilities[class_names.index(prediction)].item()
    explanation = class_info[prediction]
    
    if confidence > 0.8:
        confidence_level = "매우 높은"
    elif confidence > 0.6:
        confidence_level = "높은"
    elif confidence > 0.4:
        confidence_level = "중간"
    else:
        confidence_level = "낮은"
    
    interpretation = f"이 이미지는 {prediction}로 예측되었습니다. {explanation}\n"
    interpretation += f"모델은 이 예측에 대해 {confidence_level} 신뢰도({confidence:.2f})를 보이고 있습니다."
    
    return interpretation

# Google Drive 이미지 경로 입력 및 예측
def predict_from_drive():
    while True:
        image_path = input("Enter the path to the image in Google Drive (or 'q' to quit): ").strip()
        if image_path.lower() == 'q':
            break
        
        if not image_path:
            print("Error: Empty input. Please enter a valid path.")
            continue
        
        full_path = os.path.join('/content/drive/MyDrive/DEV/img', image_path)
        
        if not os.path.exists(full_path):
            print(f"Error: File not found at {full_path}")
            continue
        
        prediction, probabilities = predict_image(full_path)
        
        if prediction is not None and probabilities is not None:
            print(f"\nPrediction for image at {full_path}:")
            interpretation = interpret_prediction(prediction, probabilities)
            print(interpretation)
            print("\nDetailed Probabilities:")
            for class_name, prob in zip(class_names, probabilities):
                print(f"  {class_name}: {prob.item():.2f}")
            print()
        
        # GPU 메모리 정리 (필요한 경우)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# 예측 실행
print("Ready to make predictions on images from Google Drive.")
predict_from_drive()
print("Prediction completed!")
