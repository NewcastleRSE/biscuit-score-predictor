import torch.nn as nn
import torch.optim

def train(model, X_tensor, y_tensor, **kwargs):
    # Unpack kwargs
    lr = kwargs.get('lr', 0.01)
    epochs = kwargs.get('epochs', 500)
    l1_lambda = kwargs.get('l1_lambda', 0.01)

    # Set loss fn and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting Training...")

    for epoch in range(epochs):
        model.train()
        # Compute loss from model prediction
        predict = model(X_tensor)
        loss = criterion(predict, y_tensor)

        # Apply L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        # Backward pass & optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training Complete\n")

    return model