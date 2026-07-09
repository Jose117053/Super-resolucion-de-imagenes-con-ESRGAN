import time
import torch
import torch.nn.functional as F
from src.utils import rgb_to_ycbcr_batch
import src.config as cfg

def train_model(generator, discriminator, vgg_extractor, train_loader, val_loader, 
                optimizer_G, optimizer_D, bce_loss, pixel_loss, perceptual_loss, 
                device, num_epochs=150, start_epoch=0):
    '''
    Ejecuta el bucle de entrenamiento para el modelo ESRGAN.
    '''
    start_time = time.time() 

    train_pixel_losses = []
    train_percep_losses = []
    train_adv_losses   = []
    train_color_losses = []
    val_pixel_losses   = []
    val_percep_losses  = []
    val_color_losses   = []

    for epoch in range(start_epoch, num_epochs):

        epoch_start = time.time()  

        #Ponerlos en modo entrenamiento
        generator.train()
        discriminator.train()

        epoch_train_pixel = 0.0
        epoch_train_percep = 0.0
        epoch_train_adv   = 0.0
        epoch_train_color = 0.0
        train_batches = 0
        
        for i, batch in enumerate(train_loader):

            # Mover LR y HR a GPU
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            
            #Generamos imagen fake
            with torch.no_grad():
                fake_hr = generator(lr)
            
            #Calculamos probabilidad de que sea real o fake
            #==============Discriminador=========
            logits_real = discriminator(hr)                   # C(x_r)
            logits_fake = discriminator(fake_hr.detach())     # C(x_f)

            avg_fake = logits_fake.mean(dim=0, keepdim=True)  # promedio sobre batch
            avg_real = logits_real.mean(dim=0, keepdim=True)

            rel_real = logits_real - avg_fake                 # C(x_r) - E[C(x_f)]
            rel_fake = logits_fake - avg_real                 # C(x_f) - E[C(x_r)]

            
            real_tensor = torch.ones_like(rel_real, device=device) #Imagenes reales
            fake_tensor = torch.zeros_like(rel_fake, device=device) #Imagenes falsas
            
            #Error del discriminador
            loss_D_real = bce_loss(rel_real, real_tensor) #Formula de entropia cruzada binaria.
            loss_D_fake = bce_loss(rel_fake, fake_tensor)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            
            #Actualizar pesos del Discriminador
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            
            # ============Generador=============
            fake_hr = generator(lr)

            #Engañar a D haciéndole creer que fake_hr es real. Generar etiquetas de “unos” (logits_real) y comparar con la predicción de D.
            #si D cree que fake_hr es real, loss_G_adv baja, haciendo que G genere más realistas
           # pred_fake_for_G = discriminator(fake_hr)
           # real_tensor_G = torch.ones_like(pred_fake_for_G, device=device)
           # loss_G_adv = bce_loss(pred_fake_for_G, real_tensor_G) 

            logits_real = discriminator(hr)                  # C(x_r)
            logits_fake = discriminator(fake_hr)             # C(x_f)

            avg_fake = logits_fake.mean(dim=0, keepdim=True)
            avg_real = logits_real.mean(dim=0, keepdim=True)

            rel_real_G = logits_real - avg_fake           
            rel_fake_G = logits_fake - avg_real            

            loss_G_part1 = bce_loss(rel_real_G, torch.zeros_like(rel_real_G))
            loss_G_part2 = bce_loss(rel_fake_G, torch.ones_like(rel_fake_G))     
            loss_G_adv = 0.5 * (loss_G_part1 + loss_G_part2)

            #Perdida de pixel
            loss_pixel = pixel_loss(fake_hr, hr)

            #Perdida perceptual
            feats_fake = vgg_extractor(fake_hr)[0]
            feats_real = vgg_extractor(hr)[0]
            loss_percep = perceptual_loss(feats_fake, feats_real)

            #Pérdida de color
            ycbcr_fake = rgb_to_ycbcr_batch(fake_hr)
            ycbcr_real = rgb_to_ycbcr_batch(hr)
            loss_cb   = F.l1_loss(ycbcr_fake[:, 1:2, :, :], ycbcr_real[:, 1:2, :, :])
            loss_cr   = F.l1_loss(ycbcr_fake[:, 2:3, :, :], ycbcr_real[:, 2:3, :, :])
            loss_color = loss_cb + loss_cr

            #Parametros de penalizacion y relevancia
            #lambda pizel con 1.5 significa que debe ser muy fiel a los pixeles
            #lambda adv determina cuanto debe de engañar G a D, dependiendo de este, en las imagnees finales se pueden generar artefactos.
            #
            #Color asegura fidelidad en cuanto a color
            
            lambda_pixel = 1.5 #1 en un principio
            lambda_percep = 1.2 #1 en un principip
            lambda_adv = 0.005 #0.005 en un principio
            lambda_color  = 0.02 #

            #Perdida general
            loss_G = lambda_pixel * loss_pixel + lambda_percep * loss_percep + lambda_adv * loss_G_adv + lambda_color  * loss_color

            
            #Actualizar pesos del generador
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            epoch_train_pixel += loss_pixel.item()
            epoch_train_percep += loss_percep.item()
            epoch_train_adv   += loss_G_adv.item()
            epoch_train_color += loss_color.item()
            train_batches += 1



        train_pixel_losses.append(epoch_train_pixel / train_batches)
        train_percep_losses.append(epoch_train_percep / train_batches)
        train_adv_losses.append(epoch_train_adv / train_batches)
        train_color_losses.append(epoch_train_color / train_batches)

        #=======================Validacion en cada epoca
        generator.eval()
        with torch.no_grad():
            val_pixel_loss = 0.0
            val_percep_loss = 0.0
            val_color_loss  = 0.0
            count = 0
            
            for batch_val in val_loader:
                lr_val = batch_val["lr"].to(device, non_blocking=True)
                hr_val = batch_val["hr"].to(device, non_blocking=True)
                fake_hr_val = generator(lr_val)
                
                #perdida perceptual
                val_pixel_loss += pixel_loss(fake_hr_val, hr_val).item()
                feats_fake_val = vgg_extractor(fake_hr_val)[0]
                feats_real_val = vgg_extractor(hr_val)[0]
                val_percep_loss += perceptual_loss(feats_fake_val, feats_real_val).item()
                
                #Perdida de color
                ycbcr_fake_val = rgb_to_ycbcr_batch(fake_hr_val)
                ycbcr_real_val = rgb_to_ycbcr_batch(hr_val)
                cb_l = F.l1_loss(ycbcr_fake_val[:, 1:2, :, :], ycbcr_real_val[:, 1:2, :, :])
                cr_l = F.l1_loss(ycbcr_fake_val[:, 2:3, :, :], ycbcr_real_val[:, 2:3, :, :])
                val_color_loss += (cb_l + cr_l).item()

                count += 1
            
            val_pixel_loss /= count
            val_percep_loss /= count
            val_color_loss  /= count
            val_pixel_losses.append(val_pixel_loss)
            val_percep_losses.append(val_percep_loss)
            val_color_losses.append(val_color_loss)
            
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Pixel: {val_pixel_loss:.4f}, "
                f"Percep: {val_percep_loss:.4f}, "
                f"Color: {val_color_loss:.4f}, "
                f"Tiempo de época: {time.time() - epoch_start:.1f}s"
            )
        
        # ==============Guardar un Checkpoint ==============================
        # Hubo como 3 ocasiones en que estaba lluviendo y se me fue la luz, interrumpiendo el entrenamiento.
        #Pero pude recuperar el entrenamiento.
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'G_state_dict': generator.state_dict(),
                'D_state_dict': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }
            torch.save(checkpoint, f'{cfg.CHECKPOINTS_DIR}/checkpoint_epoch_{epoch+1}.pth')

    total_time = time.time() - start_time
    print(f"Entrenamiento completado en {total_time:.1f}s ({total_time/60:.1f} minutos).")
    
    # Retornamos los historiales de pérdidas por si se quieren graficar después
    return {
        "train_pixel_losses": train_pixel_losses,
        "train_percep_losses": train_percep_losses,
        "train_adv_losses": train_adv_losses,
        "train_color_losses": train_color_losses,
        "val_pixel_losses": val_pixel_losses,
        "val_percep_losses": val_percep_losses,
        "val_color_losses": val_color_losses
    }
