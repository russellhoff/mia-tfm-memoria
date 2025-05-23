# UNIR_Latex_TFG-TFM_Plantilla

<div align="left">
  <img src="https://img.shields.io/badge/Tecnolog%C3%ADa-Latex-purple">
  <img src="https://img.shields.io/badge/Trabajo-TFE-green">
  <img src="https://img.shields.io/badge/Universidad-UNIR-blue">
</div>

# Índice
* [Resumen](#resumen)
* [Instrucciones](#instrucciones)
* [Ejemplo portada](#portada)

## Resumen
Se trata de una plantilla desarrollada en LATEX que contiene el formato solicitado en los trabajos TFE de la Universidad Internacional de la Rioja (UNIR). La plantilla contiene lo siguiente:
- Portada
- Abstract en inglés y castellano
- Índice general
- Índice de figuras
- Índice de tablas
- Secciones
- Lista de acrónimos
- Bibliografía
- Anexos

![imagen](https://user-images.githubusercontent.com/46922333/211877400-e70631a7-c9a7-445e-897e-dae061d2ac20.png)

Cabe destacar también lo siguiente:

El proyecto está configurado para que sea totalmente interactivo, es decir, cada vez que referenciemos cualquier imagen, tabla, figura o sección, al seleccionarlo, este nos llevará al sitio referenciado. También mencionar que hay ejemplos de todo tipo. De esta forma, el usuario de la plantilla puede insertar cualquier tipo de elemento sin necesidad de buscar ninguna información adicional. Conseguiremos de esta forma centrar el trabajo en la realización del TFM despreocupándonos de todo lo demás.

## Instrucciones
A continuación se indican los pasos necesarios para poder trabajar con esta plantilla.
1. Deberemos ir a la página oficial de LaTeX y descargar e instalar la distribución **Tex Live**.
https://www.latex-project.org/get/#tex-distributions 
Cabe mencionar que dicha distribución solo se encuentra en entornos Linux y Windows. Los usuarios de Mac OS pueden utilizar la versión MacTex. Aunque no debería de dar ningún problema dicha versión, no se garantiza el correcto funcionamiento de la misma.
2. Aunque no es obligatorio, se recomienda instalar el entorno TeXstudio. Dicho entorno está disponible para los entornos Windows, Linux y Mac OS.
https://www.texstudio.org
3. Establecemos el documento PlantillaTFM_UNIR.tex

![imagen](https://user-images.githubusercontent.com/46922333/211872031-e802f041-922a-4b54-8830-de9e4fe6b6a5.png)

4. Seguido nos dirigimos a Opciones -> Configurar TeXstudio... -> Pestaña Compilar y configuramos las siguientes opciones.

![imagen](https://user-images.githubusercontent.com/46922333/211872426-ece1760f-bf75-4a6a-8139-fe96be650b83.png)

*Nota: Es muy importante que el compilador seleccionado sea XeLaTex y la herramienta bibliográfica BibTex.*

5. Seleccionando la flecha verde se compilará todo nuestro trabajo y nos pondrá en la raíz del proyecto un PDF con todo lo realizado.

6. En caso de que la bibliografía o la lista de acrónimos no compile, realizaremos lo siguiente:

Nos dirigimos al apartado Herramientas y ahí tendremos las opciones Bibliografía y Glosario. Seleccionaremos cada una de ellas tal como se muestra en la imagen, y seguido, volveremos a compilar nuestro proyecto.

![imagen](https://user-images.githubusercontent.com/46922333/211876604-5fa12740-f029-4710-bea3-46c69690175d.png)
![imagen](https://user-images.githubusercontent.com/46922333/211876982-24b88a81-8331-434f-9d79-71e49af2b642.png)

### Para sistemas Linux
En caso de estar en un sistema Linux tendremos que realizar un paso adicional. Copiamos la carpeta calibri, ubicada en la carpeta fonts (importante con permisos de root), a la siguiente ubicación: /usr/share/fonts/truetype

- sudo cp -r fonts/calibri  /usr/share/fonts/truetype

Seguido se actualiza la base de datos de fuentes para que el sistema reconozca las nuevas fuentes.

- sudo fc-cache -f -v

Es importante resaltar que estas fuentes son propiedad privada de Microsoft. Por lo tanto, para utilizarlas, deberíamos disponer de una licencia válida de Windows.

## Ejemplo portada <a name="portada"></a>
![imagen](https://user-images.githubusercontent.com/46922333/211885426-ae4e2c63-b494-4c5b-ab83-2c54373b9fc7.png)

