# Image Search Engine

Este proyecto es una aplicación web para buscar imágenes similares utilizando características extraídas con un modelo de red neuronal convolucional (ResNet50). El sistema permite cargar imágenes y encontrar imágenes similares en una base de datos previamente indexada.

## Estructura del Proyecto

- **app.py**: Archivo principal de la aplicación Flask que maneja las rutas y lógica del servidor.
- **image_search.py**: Contiene funciones para extraer características de imágenes y buscar imágenes similares basadas en características.
- **index.py**: Script para crear el índice de imágenes y guardar las características en un archivo `index.pkl`.
- **templates/index.html**: Plantilla HTML para la página principal donde se pueden subir imágenes y buscar por categoría.
- **templates/results.html**: Plantilla HTML para mostrar los resultados de la búsqueda.
- **static/uploads/**: Directorio para almacenar las imágenes subidas por los usuarios.
