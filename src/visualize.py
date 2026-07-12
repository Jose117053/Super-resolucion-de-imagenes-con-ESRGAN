import matplotlib.pyplot as plt

def plot_losses(historial, num_epochs):
    '''
    Toma el diccionario de historial de pérdidas del entrenamiento
    y genera las gráficas correspondientes.
    '''
    epochs = list(range(1, num_epochs + 1))
    
    # Extraemos las listas del diccionario
    train_pixel_losses = historial["train_pixel_losses"]
    val_pixel_losses = historial["val_pixel_losses"]
    train_percep_losses = historial["train_percep_losses"]
    val_percep_losses = historial["val_percep_losses"]
    train_adv_losses = historial["train_adv_losses"]
    train_color_losses = historial["train_color_losses"]
    val_color_losses = historial["val_color_losses"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_pixel_losses, label="Train Pixel")
    plt.plot(epochs, val_pixel_losses,   label="Val   Pixel")
    plt.xlabel("Época")
    plt.ylabel("Pixel Loss")
    plt.legend()
    plt.title("Pixel Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_percep_losses, label="Train Percep")
    plt.plot(epochs, val_percep_losses,   label="Val   Percep")
    
    plt.xlabel("Época")
    plt.ylabel("Perceptual Loss")
    plt.legend()
    plt.title("Perceptual Loss")

    plt.figure()
    plt.plot(epochs, train_adv_losses, label="Train Adversarial")
    plt.xlabel("Época")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.title("Adversarial Loss (Entrenamiento)")
    plt.show()

    plt.figure()
    plt.plot(epochs, train_color_losses, label="Train Color")
    plt.plot(epochs, val_color_losses,   label="Val   Color")
    plt.xlabel("Época")
    plt.ylabel("Color Loss")
    plt.legend()
    plt.title("Color Loss")
    plt.show()
