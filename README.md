# TFM – Predicción del Tráfico con Redes Neuronales y Transformers

**Autor:** Jon Inazio  

**Director:** Omar Velázquez López

**Titulación:** Máster Universitario en Inteligencia Artificial

**Universidad:** Universidad Internacional de la Rioja - UNIR 

**Curso:** 2024-2025

---

## Resumen

Este Trabajo Fin de Máster presenta el desarrollo de un modelo avanzado de predicción de tráfico basado en arquitecturas deep learning y, en concreto, en la integración de mecanismos Transformer con atención espacial. El objetivo es anticipar el flujo de vehículos en la red viaria de Bizkaia, utilizando datos abiertos de tráfico y meteorología. Se evalúan y comparan distintos modelos, destacando el enfoque Trafficformer por su capacidad para capturar correlaciones espacio-temporales complejas en escenarios reales de tráfico.

---

## Motivación

La gestión inteligente del tráfico es un reto clave para la movilidad urbana y la reducción de congestiones, emisiones y accidentes. La predicción precisa de la demanda permite optimizar infraestructuras, mejorar la toma de decisiones en tiempo real y avanzar hacia ciudades más sostenibles. La aparición de modelos basados en Transformers abre nuevas oportunidades para abordar estas tareas, superando las limitaciones de arquitecturas tradicionales como LSTM, CNN o modelos puramente estadísticos.

---

## Objetivos

- Desarrollar un sistema de predicción de tráfico que combine redes neuronales profundas y mecanismos de atención espacial tipo Transformer.
- Comparar el rendimiento de distintos modelos (MLP y Trafficformer) sobre datos reales de la red viaria de Bizkaia.
- Analizar el impacto de variables meteorológicas en la precisión de las predicciones.
- Proveer un pipeline reproducible y escalable para la adquisición, procesamiento y análisis de datos abiertos de tráfico y meteorología.

---

## Tecnologías y Herramientas

Para el desarrollo general del proyecto, se han hecho uso de las siguientes tecnologías y herramientas, entre otras:

- **Lenguajes:** Python 3.11+, Kotlin
- **Frameworks ML/DL:** PyTorch, scikit-learn
- **Gestión de experimentos:** Weights & Biases (Wandb)
- **Almacenamiento:** MongoDB, MinIO, AWS S3
- **Infraestructura:** AWS (EC2, Lambda, S3, etc.) o Local
- **Visualización:** Matplotlib, Seaborn, Folium
- **Otros:** Docker, UV, Gradle, LaTeX

---

## Licencia

Consulta el archivo [LICENSE.md](./LICENSE.md) para detalles sobre la licencia del proyecto.

---

## Contacto

**Jon Inazio**  
📧 [captain06@gmail.com](mailto:captain06@gmail.com)  
[LinkedIn](https://www.linkedin.com/in/joninazio/)