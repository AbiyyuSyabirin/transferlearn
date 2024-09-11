# Load the trained model weights with map_location for CPU compatibility
def get_model(num_classes):
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)

    # Change the fully connected layer for the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Return the model
    return model

# Load model state_dict
def load_model(path, num_classes):
    model = get_model(num_classes)
    
    # Load the entire dictionary with CPU map location
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # Extract and load the model state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("The checkpoint does not contain a model_state_dict key")
    
    model.eval()
    return model

# Number of classes in the dataset
NUM_CLASSES = 2
model = load_model('models/best_model_params.pt', NUM_CLASSES)

# The class names of the model
CLASS_NAMES = ['cats', 'dogs']

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    token: str = Depends(authorize)  # Authorization check
) -> Dict[str, List[Dict[str, str]]]:
    try:
        # Read the image file
        image, img_size = read_file_as_image(await file.read())
        
        # Convert image to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor and reorder dimensions
        
        # Move the image and model to the appropriate device
        if torch.cuda.is_available():
            image_tensor = image_tensor.to('cuda')
            model.to('cuda')
        else:
            image_tensor = image_tensor.to('cpu')
            model.to('cpu')
        
        # Perform prediction
        with torch.no_grad():
            model.eval()
            outputs = model(image_tensor)
            predictions = torch.exp(outputs)

            # Debugging information
            print("Predictions tensor shape:", predictions.shape)
            print("Predictions tensor:", predictions)
            
            # Ensure topk is within bounds
            topk = min(3, NUM_CLASSES)  # Ensure topk is not greater than number of classes
            topk_values, topk_indices = torch.topk(predictions, topk, dim=1)
            
            # Ensure the output format
            topk_classes = [CLASS_NAMES[i] for i in topk_indices[0].tolist()]
            topk_scores = topk_values[0].tolist()
            
            results = [
                {"class": topk_classes[i], "score": float(topk_scores[i])}
                for i in range(topk)
            ]

        # Return predictions
        return JSONResponse(content={
            'predictions': results
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
