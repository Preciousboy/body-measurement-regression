import torch

def validating_loop(
        model,
        dataloader,
        loss_fn,
        device
):
    model = model.to(device)
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for image, targets, _, _ in dataloader:

            image = image.to(device)
            targets = targets.to(device)

            prediction = model(image)

            loss = loss_fn(prediction, targets)

            total_loss += (
                loss.item()
                * image.size(0)
                )

    val_loss = total_loss / len(dataloader.dataset)
    return val_loss