# Sistema de Búsqueda de Imágenes Similares

Este proyecto es una aplicación web desarrollada en Flask para la búsqueda de imágenes similares utilizando características extraídas con un modelo de red neuronal convolucional (ResNet50). La aplicación permite a los usuarios cargar imágenes y encontrar imágenes similares en una base de datos previamente indexada.

## Estructura del Proyecto

- **`app.py`**: Archivo principal de la aplicación Flask que maneja las rutas y la lógica del servidor, gestionando la carga de imágenes y la búsqueda de similitudes.
- **`image_search.py`**: Contiene funciones para extraer características de las imágenes utilizando ResNet50 y para buscar imágenes similares en la base de datos basada en esas características.
- **`index.py`**: Script para crear un índice de imágenes y guardar las características extraídas en un archivo `index.pkl`, facilitando búsquedas rápidas.
- **`templates/index.html`**: Plantilla HTML para la página principal de la aplicación, donde los usuarios pueden subir imágenes y realizar búsquedas por categoría.
- **`templates/results.html`**: Plantilla HTML para mostrar los resultados de las búsquedas de imágenes similares.
- **`static/uploads/`**: Directorio destinado a almacenar las imágenes subidas por los usuarios, permitiendo su procesamiento y búsqueda.

## Cómo Ejecutar el Proyecto

1. Clona el repositorio:
    ```bash
    git clone <URL del repositorio>
    ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

3. Ejecuta la aplicación:
    ```bash
    python app.py
    ```

4. Accede a la aplicación en tu navegador en `http://127.0.0.1:5000`.

## Tecnologías Utilizadas

- Flask
- TensorFlow (ResNet50)
- NumPy
- scikit-learn


