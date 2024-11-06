from ultralytics import YOLO
import torch
import os

def main_worker():
    # Set device to GPU 2 or 3
    device_id = 2  # or 3
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with configuration file
    model = YOLO("yolov8x.yaml")

    # Load pretrained weights
    state_dict = torch.load("/work/cshah/updatedYOLOv8/ultralytics/runs/detect/train74/weights/best.pt")
    model_state_dict = {k: v for k, v in state_dict.items() if k.startswith('model.model.')}
    model.model.load_state_dict(model_state_dict, strict=False)

    # Set model to training mode and move to GPU
    model.model.train()
    model.model.to(device)

    # Training configuration
    epochs = 500
    patience = 10  # Early stopping patience
    best_val_loss = float('inf')
    trigger_times = 0

    # Training loop
    for epoch in range(epochs):
        # Train the model
        model.train(data="pasca_data_GFISHERD24.yaml", epochs=1)  # Train for one epoch in each iteration

        # Validate the model
        model.model.eval()
        with torch.no_grad():
            metrics = model.val()
            val_loss = metrics['loss'] if 'loss' in metrics else float('inf')  # Extract loss from metrics

        # Print metrics
        print(f"Epoch {epoch + 1}/{epochs}: Validation Loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model checkpoint
            torch.save(model.model.state_dict(), 'best_model.pt')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping triggered')
                break

    # Final testing
    model.model.eval()
    with torch.no_grad():
        test_metrics = model.val(data='pasca_data_GFISHERD24.yaml', device=2,split='test',save_json=True)
        print("Test Metrics:", test_metrics)

if __name__ == '__main__':
    main_worker()
