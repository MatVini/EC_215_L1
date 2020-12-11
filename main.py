from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import math


def dnorm(x, mu, sd):  # Distribuição normal
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1d[i] = dnorm(kernel_1d[i], 0, sigma)
    kernel_2d = np.outer(kernel_1d.T, kernel_1d.T)

    kernel_2d *= 1.0 / kernel_2d.max()

    if verbose:
        plt.imshow(kernel_2d, interpolation='none', cmap='gray')
        plt.title('Kernel')
        plt.show()

    return kernel_2d


def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:  # Se imagem for colorida, transforma em grayscale
        image = ((0.2126 * image[:, :, 0]) + (0.7152 * image[:, :, 1]) + (0.0722 * image[:, :, 2])).astype(np.uint8)

    image_row, image_col = image.shape  # Pega tamanho da image
    kernel_row, kernel_col = kernel.shape  # Pega tamanho do kernel

    output = np.zeros(image.shape)  # Prepara imagem final

    pad_height = int((kernel_row - 1) / 2)  # Bordas verticais
    pad_width = int((kernel_col - 1) / 2)  # e horizontais para a imagem final

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))  # Prepara o frame para a imagem final

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image  # Coloca a imagem final no frame

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col]) # Passa o kernel pela imagem (
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1] # Alisa a imagem

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title('Imagem com ruído reduzido')
        plt.show()

    return output


def gaussian_blur(img, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=int(math.sqrt(kernel_size)), verbose=verbose)
    return convolution(img, kernel, average=True, verbose=verbose)


# ==================================================================================================================== #

image_1 = np.array(Image.open('kodim.jpg'))  # Abre a imagem (deve ser JPEG ou PNG)

image_1_dn = gaussian_blur(image_1, 7, verbose=True)

image_1_dn = Image.fromarray(image_1_dn)
if image_1_dn.mode != 'L':
    image_1_dn = image_1_dn.convert('L')  # Converte para grayscale para salvar
    image_1_dn = ImageEnhance.Brightness(image_1_dn).enhance(2.25)  # Aumenta o brilho porque a função anterior escurece a imagem

image_1_dn.save('kodim_dn.jpg')
