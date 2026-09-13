import numpy as np
import matplotlib.pyplot as plt
from sat_images_proccesing_functions import *

# Prueba 1: Cargar dos imagenes B04y B08 y calcular el NDVI
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))

# Imagen satelital Julio 2020 B04 Raw
img1 = ImagenSatelital("images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff")
rgb1 = img1.obtener_imagen()
#img1.info()

ax1.imshow(rgb1)
ax1.set_title("Imagen 2020 B04")
ax1.axis("off")

# Imagen satelital Julio 2020 B08 Raw
img2 = ImagenSatelital("images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B08_(Raw).tiff")
rgb2 = img2.obtener_imagen()
#img2.info()

ax2.imshow(rgb2)
ax2.set_title("Imagen 2020 B08")
ax2.axis("off")

# NDVI
ndvi = NDVICalculo(img1.ruta_imagen, img2.ruta_imagen)
img_ndvi = ndvi.obtener_ndvi()

ax3.imshow(img_ndvi)
ax3.set_title("NDVI")
ax3.axis("off")

plt.tight_layout()
plt.show()


# Prueba 2: Cargar dos imagenes NDVI de 2020 y 2024 y calcular la diferencia

# Mostrar los cuatro plots en un solo gráfico
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 12))

# Imagen satelital Julio 2020
img1 = ImagenSatelital("images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_NDVI.tiff")
rgb1 = img1.obtener_imagen()
#img1.info()

axs[0,0].imshow(rgb1)
axs[0,0].set_title("Imagen Satelital Original Julio 2020")
axs[0,0].axis("off")

# Imagen satelital Julio 2024
img2 = ImagenSatelital("images/2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_NDVI.tiff")
rgb2 = img2.obtener_imagen()
#img2.info()

axs[1,0].imshow(rgb2)
axs[1,0].set_title("Imagen Satelital Original Julio 2024")
axs[1,0].axis("off")

# Diferencia entre NDVIs
analisis = NDVICalculo(img1.ruta_imagen, img2.ruta_imagen)
diferencia_img = analisis.NDVI_diferencia_temporal()

axs[2,0].imshow(diferencia_img, cmap="gray")
axs[2,0].set_title("Diferencia de NDVI")
axs[2,0].axis("off")

# NDVI1
ndvi1 = NDVICalculo("images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff", "images/2020-07-01-00_00_2020-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B08_(Raw).tiff")
ndvi_img1 = ndvi1.obtener_ndvi()

axs[0,1].imshow((ndvi_img1 * 255).astype(np.uint8))
axs[0,1].set_title("NDVI1")
axs[0,1].axis("off")

# NDVI2
ndvi2 = NDVICalculo("images/2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B04_(Raw).tiff", "images/2024-07-01-00_00_2024-07-01-23_59_Sentinel-2_Quarterly_Mosaics_B08_(Raw).tiff")
ndvi_img2 = ndvi2.obtener_ndvi()

axs[1,1].imshow((ndvi_img2 * 255).astype(np.uint8))
axs[1,1].set_title("NDVI2")
axs[1,1].axis("off")

# Ajustar y mostrar
plt.tight_layout()
plt.show()
