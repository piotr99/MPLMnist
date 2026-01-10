from torchvision import datasets, transforms
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
import torch
import torch_directml
from torchvision import datasets, transforms

datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

device = torch_directml.device()

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images
def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

try:
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    X_test = load_mnist_images(f"{curr_dir}\\data\\MNIST\\raw\\t10k-images-idx3-ubyte")
    y_test = load_mnist_labels(f"{curr_dir}\\data\\MNIST\\raw\\t10k-labels-idx1-ubyte")
except:
    print("Pliki nie istnieją")



# 1. Funkcje pomocnicze
def get_one_hot(targets, num_classes=10):
    res = np.eye(num_classes)[targets].T
    return res

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# 2. Parametry i inicjalizacja
input_size = 784
hidden_size = 16384
output_size = 10

W1 = (torch.randn(hidden_size, input_size, device=device) * torch.sqrt(torch.tensor(2. / input_size))).requires_grad_(False)
b1 = torch.zeros(hidden_size, 1, device=device)
W2 = (torch.randn(output_size, hidden_size, device=device) * torch.sqrt(torch.tensor(2. / hidden_size))).requires_grad_(False)
b2 = torch.zeros(output_size, 1, device=device)

learning_rate = 0.05
epochs = 5

    # 3. Pętla treningowa
for epoch in range(epochs):
    print(f"\n--- EPOKA {epoch + 1} ---")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Przygotowanie batcha: spłaszczenie obrazu (28x28 -> 784)
        X_batch = data.view(-1, 784).to(device).t() # Macierz (784, 32)
        
        # One-hot encoding dla targetu
        Y_batch = torch.zeros(output_size, X_batch.shape[1], device=device)
        Y_batch.scatter_(0, target.to(device).unsqueeze(0), 1)
        
        m = X_batch.shape[1]

        # --- FORWARD PROPAGATION ---
        Z1 = torch.mm(W1, X_batch) + b1
        A1 = torch.relu(Z1)

        Z2 = torch.mm(W2, A1) + b2
        # Softmax stabilny numerycznie
        A2 = torch.softmax(Z2, dim=0)

        # --- BACKPROPAGATION ---
        dZ2 = A2 - Y_batch
        dW2 = (1/m) * torch.mm(dZ2, A1.t())
        db2 = (1/m) * torch.sum(dZ2, dim=1, keepdim=True)

        dZ1 = torch.mm(W2.t(), dZ2) * (Z1 > 0).float()
        dW1 = (1/m) * torch.mm(dZ1, X_batch.t())
        db1 = (1/m) * torch.sum(dZ1, dim=1, keepdim=True)

        # --- AKTUALIZACJA ---
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if batch_idx % 500 == 0:
            predictions = torch.argmax(A2, dim=0)
            accuracy = (predictions == target.to(device)).float().mean()
            print(f"Batch {batch_idx}, Accuracy: {accuracy*100:.2f}%")

# ==============
# Test
# ==============
# Przygotowanie danych testowych na GPU
X_test_gpu = torch.tensor(X_test, dtype=torch.float32, device=device).view(-1, 784).t() / 255.0
y_test_gpu = torch.tensor(y_test, device=device)

def check_test_accuracy():
    # Forward pass na całym zbiorze testowym naraz
    Z1 = torch.mm(W1, X_test_gpu) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.mm(W2, A1) + b2
    A2 = torch.softmax(Z2, dim=0)

    predictions = torch.argmax(A2, dim=0)
    correct = (predictions == y_test_gpu).sum().item()
    total = y_test_gpu.size(0)
    accuracy = (correct / total) * 100
    
    print(f"Test: {correct} poprawnych, {total - correct} błędnych → {accuracy:.2f}%")

check_test_accuracy()









# ==============================================================================
# CZĘŚĆ 1: Funkcja predykcji korzystająca z Twojej sieci
# ==============================================================================
# Zakładamy, że zmienne W1, b1, W2, b2 istnieją globalnie po treningu.
def center_image(img_tensor_28x28):
    # Znajdź niezerowe piksele (maska > 0.01)
    # torch.nonzero zwraca indeksy [wiersz, kolumna]
    coords = torch.nonzero(img_tensor_28x28 > 0.01)

    if coords.size(0) == 0:
        return torch.zeros((28, 28), device=device)

    # Oblicz granice (bounding box)
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values

    digit_crop = img_tensor_28x28[y_min:y_max+1, x_min:x_max+1]
    h, w = digit_crop.shape

    # Oblicz offset do wycentrowania
    offset_y = (28 - h) // 2
    offset_x = (28 - w) // 2

    centered_img = torch.zeros((28, 28), device=device)
    centered_img[offset_y:offset_y+h, offset_x:offset_x+w] = digit_crop
    
    return centered_img

def make_prediction(img_array_28x28):
    # Konwersja wejścia z GUI (NumPy) na Tensor na GPU
    img_tensor = torch.tensor(img_array_28x28, dtype=torch.float32, device=device)
    
    # Inwersja i normalizacja (MNIST: czarne tło, biała cyfra)
    img_tensor = (255.0 - img_tensor) / 255.0
    
    # Centrowanie
    centered_tensor = center_image(img_tensor)
    
    # Przygotowanie do sieci (spłaszczenie do 784, 1)
    X_input = centered_tensor.view(784, 1)

    # Forward Propagation
    Z1 = torch.mm(W1, X_input) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.mm(W2, A1) + b2
    A2 = torch.softmax(Z2, dim=0)

    # Wynik
    prediction = torch.argmax(A2).item()
    probability = A2[prediction].item()
    
    return prediction, probability

# ==============================================================================
# CZĘŚĆ 2: Aplikacja GUI do rysowania (Tkinter + PIL)
# ==============================================================================

class DigitDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Narysuj cyfrę (MNIST)")

        # Konfiguracja płótna
        self.canvas_size = 280  # Rysujemy na większym, potem zmniejszymy
        self.brush_size = 20    # Grubość pędzla
        
        # Tkinter Canvas - to co widzi użytkownik
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white', bd=3, relief="ridge")
        self.canvas.pack(pady=10)

        # PIL Image - to co dzieje się w pamięci (niewidoczne tło do zapisu)
        # Tworzymy biały obrazek "L" (skala szarości)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw_handle = ImageDraw.Draw(self.image)

        # Rysowanie (przytrzymanie lewego przycisku myszy)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Przyciski
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_predict = tk.Button(btn_frame, text="Zgadnij!", command=self.predict_digit, bg="#ddffdd", font=("Arial", 12))
        self.btn_predict.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(btn_frame, text="Wyczyść", command=self.clear_canvas, font=("Arial", 12))
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        # Etykieta wyniku
        self.result_label = tk.Label(root, text="Narysuj cyfrę i kliknij Zgadnij!", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def paint(self, event):
        """Rysuje czarne kółka tam gdzie jest myszka"""
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        
        # Rysuj na widocznym Canvasie Tkintera
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        
        # Rysuj na niewidocznym obrazku PIL w pamięci
        self.draw_handle.ellipse([x1, y1, x2, y2], fill=0, outline=0)

    def clear_canvas(self):
        """Czyści płótno i obraz w pamięci"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw_handle = ImageDraw.Draw(self.image)
        self.result_label.config(text="Wyczyszczono.")

    def predict_digit(self):
        """Pobiera obraz, przetwarza go i wysyła do sieci"""
        # Zmiana rozmiaru na 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Konwersja na tablicę NumPy (którą potem make_prediction zamieni na tensor)
        img_array = np.array(img_resized, dtype=np.uint8)
        
        try:
            digit, prob = make_prediction(img_array)
            self.result_label.config(text=f"Widzę cyfrę: {digit} (Pewność: {prob*100:.1f}%)", fg="black")
        except Exception as e:
            self.result_label.config(text=f"Błąd: {str(e)}", fg="red")
            print(f"Error: {e}")
# ==============================================================================
# Uruchomienie aplikacji
# ==============================================================================

# Upewnijmy się, że wagi istnieją przed startem GUI (proste sprawdzenie)
if 'W1' in globals() and 'b2' in globals():
    print("Sieć wytrenowana. Uruchamiam GUI...")
    root = tk.Tk()
    app = DigitDrawerApp(root)
    # Zatrzymuje działanie skryptu i wyświetla okno
    root.mainloop()
else:
    print("BŁĄD: Najpierw musisz wytrenować sieć (uruchom pętlę for)!")

    print("Zmienne W1, b1, W2, b2 nie istnieją w pamięci.")
