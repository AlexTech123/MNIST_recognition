import torch

def predict(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in data_loader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions