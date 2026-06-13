import sys
import yaml
import torch
import traceback
from src.dataset import build_dataloaders
from src.model import build_model, build_criterion

def test():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    train_loader, _, _ = build_dataloaders(cfg)
    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    from torchmetrics.classification import MulticlassF1Score
    f1_metric = MulticlassF1Score(num_classes=10, average="macro").to(device)

    print("Iterating...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}")
        images = images.to(device)
        labels = labels.to(device)

        print("Forward...")
        logits = model(images)
        loss = criterion(logits, labels)

        print("Backward...")
        optimizer.zero_grad()
        loss.backward()
        
        print("Step...")
        optimizer.step()

        print("Metrics...")
        preds = logits.argmax(dim=1)
        hard_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
        f1_metric.update(preds, hard_labels)

        print("Done batch.")
        break
    
    print(f"F1: {f1_metric.compute().item()}")

if __name__ == "__main__":
    try:
        test()
        print("Success!")
    except Exception as e:
        traceback.print_exc()
