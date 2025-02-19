import sys
import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from torchsummary import summary
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QLabel, QFileDialog, QMessageBox, QDialog, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

class IntegratedClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Classifier (MNIST & Cat/Dog)")
        self.setGeometry(100, 100, 1200, 800)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mnist_model = self.load_mnist_model()
        self.catdog_model = self.load_catdog_model()
        self.loaded_mnist_image = None
        self.loaded_catdog_image = None
        self.create_widgets()

    def load_mnist_model(self):
        model = models.vgg16_bn(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model = model.to(self.device)
        try:
            model.load_state_dict(torch.load('model/vgg16_mnist.pth'))
            model.eval()
        except:
            QMessageBox.warning(self, "Warning", "No MNIST pre-trained model found.")
        return model

    def load_catdog_model(self):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(self.device)
        try:
            model.load_state_dict(torch.load('model/resnet50_with_random_erasing.pth'))
            model.eval()
        except:
            QMessageBox.warning(self, "Warning", "No Cat/Dog pre-trained model found.")
        return model

    def create_widgets(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top section (MNIST)
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        
        mnist_container = QWidget()
        mnist_layout = QHBoxLayout(mnist_container)

        # MNIST buttons
        mnist_buttons = QWidget()
        mnist_buttons_layout = QVBoxLayout(mnist_buttons)
        
        self.show_mnist_structure_btn = QPushButton("1.1 Show Structure")
        self.show_mnist_results_btn = QPushButton("1.2 Show Acc and Loss")
        self.load_mnist_image_btn = QPushButton("Load Image")
        self.predict_mnist_btn = QPushButton("1.3 Predict")

        for btn in [self.load_mnist_image_btn, self.show_mnist_structure_btn, self.show_mnist_results_btn,
                   self.predict_mnist_btn]:
            mnist_buttons_layout.addWidget(btn)

        # MNIST image display
        self.mnist_image_label = QLabel()
        self.mnist_image_label.setAlignment(Qt.AlignCenter)
        self.mnist_image_label.setMinimumSize(300, 300)
        self.mnist_image_label.setStyleSheet("QLabel { background-color: white; border: 1px solid gray; }")

        mnist_layout.addWidget(mnist_buttons)
        mnist_layout.addWidget(self.mnist_image_label)
        
        # Right section (Cat/Dog)
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)

        catdog_container = QWidget()
        catdog_layout = QHBoxLayout(catdog_container)

        # Cat/Dog buttons
        catdog_buttons = QWidget()
        catdog_buttons_layout = QVBoxLayout(catdog_buttons)

        self.load_catdog_image_btn = QPushButton("Q2 Load Image")
        self.show_catdog_structure_btn = QPushButton("2.2 Show Model Structure")
        self.show_catdog_results_btn = QPushButton("2.3 Show Comparison")
        self.predict_catdog_btn = QPushButton("2.4 Inference")
        self.show_catdog_images_btn = QPushButton("2.1 Show Image")

        for btn in [self.load_catdog_image_btn, self.show_catdog_images_btn, self.show_catdog_structure_btn, self.show_catdog_results_btn,
                   self.predict_catdog_btn]:
            catdog_buttons_layout.addWidget(btn)

        # Cat/Dog image display
        self.catdog_image_label = QLabel()
        self.catdog_image_label.setAlignment(Qt.AlignCenter)
        self.catdog_image_label.setMinimumSize(300, 300)
        self.catdog_image_label.setStyleSheet("QLabel { background-color: white; border: 1px solid gray; }")

        catdog_layout.addWidget(catdog_buttons)
        catdog_layout.addWidget(self.catdog_image_label)

        # Add containers to layouts
        top_layout.addWidget(mnist_container)
        bottom_layout.addWidget(catdog_container)

        # Result labels
        self.mnist_result_label = QLabel()
        self.mnist_result_label.setAlignment(Qt.AlignCenter)
        self.mnist_result_label.setStyleSheet("QLabel { font-size: 18pt; }")
        top_layout.addWidget(self.mnist_result_label)

        self.catdog_result_label = QLabel()
        self.catdog_result_label.setAlignment(Qt.AlignCenter)
        self.catdog_result_label.setStyleSheet("QLabel { font-size: 18pt; }")
        bottom_layout.addWidget(self.catdog_result_label)

        # Add sections to main layout
        main_layout.addWidget(top_section)
        main_layout.addWidget(bottom_section)

        self.connect_buttons()

    def connect_buttons(self):
        self.show_mnist_structure_btn.clicked.connect(self.show_mnist_structure)
        self.show_mnist_results_btn.clicked.connect(self.show_mnist_training_results)
        self.load_mnist_image_btn.clicked.connect(self.load_mnist_image)
        self.predict_mnist_btn.clicked.connect(self.predict_mnist)

        self.show_catdog_structure_btn.clicked.connect(self.show_catdog_structure)
        self.show_catdog_results_btn.clicked.connect(self.show_catdog_comparison)
        self.load_catdog_image_btn.clicked.connect(self.load_catdog_image)
        self.predict_catdog_btn.clicked.connect(self.predict_catdog)
        self.show_catdog_images_btn.clicked.connect(self.show_catdog_images)

    def pil_to_pixmap(self, pil_image):
        if pil_image.mode == "L":
            img = pil_image.convert("L")
            h, w = img.size
            data = img.tobytes()
            qimg = QImage(data, w, h, w, QImage.Format_Grayscale8)
        else:
            img = pil_image.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.size[0], img.size[1], img.size[0] * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def show_mnist_structure(self):
        if self.mnist_model:
            summary(self.mnist_model, (1, 32, 32))

    def show_mnist_training_results(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("MNIST Training Results")
        dialog.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout(dialog)
        label = QLabel(dialog)
        try:
            pixmap = QPixmap('analysis/training_results.png')
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
        except:
            label.setText("MNIST training results not found.")
            label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        dialog.exec_()

    def load_mnist_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open MNIST Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.loaded_mnist_image = file_path
            image = Image.open(file_path).convert('L').resize((224, 224))
            pixmap = self.pil_to_pixmap(image)
            self.mnist_image_label.setPixmap(pixmap)

    def load_catdog_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Cat/Dog Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.loaded_catdog_image = file_path
            image = Image.open(file_path).convert('RGB').resize((224, 224))
            pixmap = self.pil_to_pixmap(image)
            self.catdog_image_label.setPixmap(pixmap)

    def predict_mnist(self):
        if not self.loaded_mnist_image:
            QMessageBox.warning(self, "Warning", "No MNIST image loaded.")
            return

        image = Image.open(self.loaded_mnist_image).convert('L')
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.mnist_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = outputs.argmax(1).item()
        
        self.mnist_result_label.setText(f"MNIST Prediction: {predicted_class}")
        self.show_probability_distribution_window(probabilities.cpu().numpy())

    def predict_catdog(self):
        if not self.loaded_catdog_image:
            QMessageBox.warning(self, "Warning", "No Cat/Dog image loaded.")
            return

        image = Image.open(self.loaded_catdog_image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.catdog_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_names = ['Cat', 'Dog']
            result = class_names[predicted.item()]
        self.catdog_result_label.setText(f"Cat/Dog Prediction: {result}")

    def show_catdog_structure(self):
        if self.catdog_model:
            summary(self.catdog_model, (3, 224, 224))

    def show_catdog_comparison(self):
        try:
            img = plt.imread('analysis/accuracy_comparison.png')
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except:
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("Cat/Dog accuracy comparison not found.")
            error_dialog.exec_()

    def show_catdog_images(self):
        dataset_dir = 'inference_dataset'
        if not os.path.exists(dataset_dir):
            self.catdog_result_label.setText("Error: Dataset folder not found.")
            return

        classes = ['Cat', 'Dog']
        images = {}
        for class_name in classes:
            class_dir = os.path.join(dataset_dir, class_name.lower())
            if not os.path.exists(class_dir):
                self.catdog_result_label.setText(f"Error: Class folder '{class_name}' not found.")
                return

            image_files = [f for f in os.listdir(class_dir) 
                         if os.path.isfile(os.path.join(class_dir, f))]
            if image_files:
                random_image = random.choice(image_files)
                images[class_name] = os.path.join(class_dir, random_image)

        if images:
            fig, axes = plt.subplots(1, len(images), figsize=(8, 4))
            transform = transforms.Compose([transforms.Resize((224, 224))])
            for ax, (class_name, image_path) in zip(axes, images.items()):
                img = Image.open(image_path)
                img = transform(img)
                ax.imshow(img)
                ax.set_title(class_name)
                ax.axis('off')
            plt.show()

    def show_probability_distribution_window(self, probabilities):
        dialog = QDialog(self)
        dialog.setWindowTitle("Probability Distribution")
        dialog.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout(dialog)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(probabilities)), probabilities)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title('Probability of each class')

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()

def main():
    app = QApplication(sys.argv)
    window = IntegratedClassifierGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()