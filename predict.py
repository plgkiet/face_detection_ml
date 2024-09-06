import torch
from model import Model  # Import your model class
from torchvision import transforms
from PIL import Image

# Load the saved model
input_size = 300 * 300 * 3  # Since your images have 4 channels (300x300x4)
hidden_size = 128
output_size = 2
model = Model(input_size, hidden_size, output_size)

model.load_state_dict(torch.load("main_program/model.pth"))  # Load the trained model
model.eval()  # Set the model to evaluation mode

# Define the same transformations as during training
transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Assuming 4 channels
    ]
)


def predict_image(image_path):
    # Open the image
    img = Image.open(image_path).convert("RGB")  # Ensure it's RGBA

    # Preprocess the image
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img)

    # Get the predicted class (0 or 1)
    predicted_class = torch.argmax(output, dim=1).item()

    # Map the prediction to the actual class
    if predicted_class == 1:
        print("Predicted: Robert Downey Jr")
    else:
        print("Predicted: Megan Fox")


image_path = "main_program/test/megan_fox_01.png"
predict_image(image_path)