import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio.plot import reshape_as_image

# Clase base para representar imágenes satelitales
class ImagenSatelital:
    def __init__(self, ruta_imagen):
        self.ruta_imagen = ruta_imagen

        with rasterio.open(ruta_imagen) as src:
            self.datos = src.read()             # (bandas, alto, ancho)
            self.perfil = src.profile           # Metadatos GeoTIFF completos
            self.crs = src.crs                  # Sistema de coordenadas (EPSG)
            self.transform = src.transform      # Geotransformación (resolución + origen)
            self.bounds   = src.bounds          # Bounding box geográfico
            self.nodata   = src.nodata          # Valor de "sin dato"
            self.n_bandas = src.count           # Número de bandas 

    def info(self):
        """Imprime un resumen de los metadatos de la imagen."""
        alto, ancho = self.tamaño_imagen()
        res_x, res_y = self.resolucion_espacial()
        print(f"Archivo   : {self.ruta_imagen}")
        print(f"Tamaño    : {ancho} x {alto} px")
        print(f"Bandas    : {self.n_bandas}")
        print(f"Resolución: {res_x:.2f} x {res_y:.2f} m/px")
        print(f"CRS       : {self.crs}")
        print(f"Bounds    : {self.bounds}")
        print(f"NoData    : {self.nodata}")
        print(f"Dtype     : {self.perfil['dtype']}")

    def tamaño_imagen(self):
        """Devuelve (alto, ancho) en píxeles."""
        _, alto, ancho = self.datos.shape
        return alto, ancho
    
    def resolucion_espacial(self):
        """Devuelve la resolución en unidades del CRS (metros si es UTM)."""
        return self.transform.a, abs(self.transform.e)  # (ancho_pixel, alto_pixel)

    def obtener_imagen(self):
        """
        Devuelve imagen RGB normalizada para visualización con matplotlib (contraste via percentil).
        Si tiene más de 3 bandas, usa las 3 primeras (R, G, B).
        Si solo tiene 1 banda, la replica en los 3 canales.
        """
        if self.n_bandas >= 3:
            rgb = self.datos[:3, :, :]              # Primeras 3 bandas
        else:
            rgb = np.repeat(self.datos[:1], 3, axis=0)  # Escala de grises → RGB

        # Mover ejes: (bandas, alto, ancho) → (alto, ancho, bandas)
        rgb = reshape_as_image(rgb).astype(float)

        # Enmascarar nodata si existe
        if self.nodata is not None:
            rgb[rgb == self.nodata] = np.nan

        vmin = np.nanpercentile(rgb, 2) # tomamos el valor minimo en percentil 2
        vmax = np.nanpercentile(rgb, 98) # tomamos el valor maximo en percentil 98

        if vmax == vmin:
            return np.zeros_like(rgb, dtype=np.uint8)

        rgb_clip = np.clip(rgb, vmin, vmax) # recorta outliers fuera del rango
        rgb_norm = ((rgb_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        return rgb_norm

    def obtener_banda(self, numero_banda):
        """
        Devuelve una banda específica como array 2D.
        numero_banda: 1-indexado (igual que rasterio)
        """
        if numero_banda < 1 or numero_banda > self.n_bandas:
            raise ValueError(f"Banda {numero_banda} no existe. "
                                f"Esta imagen tiene {self.n_bandas} bandas.")
        banda = self.datos[numero_banda - 1].astype(float)

        # Enmascarar nodata
        if self.nodata is not None:
            banda[banda == self.nodata] = np.nan

        return banda

# Clase base para calcular NDVI a partir de imagenes satelitales
class NDVICalculo:
    def __init__(self, imagen_banda_roja, imagen_banda_infrarroja):
        self.imagen_banda_roja = imagen_banda_roja              # B04
        self.imagen_banda_infrarroja = imagen_banda_infrarroja  # B08

        with rasterio.open(self.imagen_banda_roja) as src:
            self.banda_roja = src.read(1)  # Lee la primera banda
        with rasterio.open(self.imagen_banda_infrarroja) as src:
            self.banda_infrarroja = src.read(1)  # Lee la primera banda
        return
    
    def obtener_ndvi(self):
        """Calcula el NDVI (Índice de Vegetación de Diferencia Normalizada)."""
        denominador = self.banda_infrarroja + self.banda_roja

        # Calculo de NDVI
        img_ndvi = np.where(denominador == 0, 0.0,                                     # Si denominador=0, NDVI=0
                       (self.banda_infrarroja - self.banda_roja) / denominador         # Si no, calcular normal
        )
        
        # Recortar valores al rango válido [-1, 1]
        img_ndvi = np.clip(img_ndvi, -1, 1)
        
        return img_ndvi

    def NDVI_diferencia_temporal(self):
        """Calcula la diferencia entre dos imágenes satelitales de un mismo
        lugar en dos epocas diferentes."""
        return cv.absdiff(self.banda_roja, self.banda_infrarroja)

    
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
