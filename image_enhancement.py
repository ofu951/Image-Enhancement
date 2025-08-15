import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model('best_model.h5', compile=False)

try:
    model.summary()
    print("Model successfully uploaded")
except Exception as e:
    print(f"Error: {e}")


image_path = 'images/frame_2262.jpg'  # Your picture path
image = cv2.imread(image_path)

# Görseli BGR'den RGB'ye çevir
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


input_image = cv2.resize(image_rgb, (640, 360))  # Model giriş boyutuna göre ayarla
input_image = np.expand_dims(input_image, axis=0)  # Batch boyutu ekle
input_image = input_image / 255.0  # Normalizasyon

# Görseli modelle işleyin (enhancement yapın)
enhanced_image = model.predict(input_image)

# Çıktıyı görselleştirilebilir formata çevir
enhanced_image = (enhanced_image[0] * 255.0).astype(np.uint8)
enhanced_resized_image = cv2.resize(enhanced_image, (1280, 720))  # Model giriş boyutuna göre ayarla

# Kayıt klasörünü oluştur
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Görseli kaydet
output_path = os.path.join(output_dir, "enhanced_25.jpg")
cv2.imwrite(output_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))  

print(f"Saved: {output_path}")


plt.figure(figsize=(15, 9))


plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Geliştirilmiş görsel
plt.subplot(1, 2, 2)
plt.imshow(enhanced_image)
plt.title('Enhanced Image')
plt.axis('off')

plt.show()

