def training_loop(model, 
                  optimizer, 
                  loss_fn,
                  dataloader,
                  device):

    """
    Train the model for one complete epoch.

    Returns:
        training loss
    """
    model = model.to(device)
    model.train()

    total_loss = 0.0

    for image, target, _, _ in dataloader:


        # Choosing our device
        image = image.to(device)
        target = target.to(device)

        # Clearing old gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(image)

        # Calculate loss
        loss = loss_fn(predictions, target)

        # Backward pass
        loss.backward()

        # Updating model parameters
        optimizer.step()

        # Accum
        total_loss += (
                        loss.item() 
                        * image.size(0)
                        )

    training_loss = total_loss / len(dataloader.dataset)
    return training_loss