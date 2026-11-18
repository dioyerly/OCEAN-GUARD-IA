# OCEAN GUARD AI · Ocean Watch (demostración académica)

Plataforma para analizar imágenes satelitales y señalar **posibles** derrames de petróleo con una U-Net + ResNet34.
El sistema sugiere, no confirma: toda detección requiere revisión humana.

## Iniciar

```powershell
cd C:\Users\ANDAMIOS\Desktop\OCEAN-GUARD-AI\platform
.\iniciar_plataforma.ps1
```

**Reiniciar tras una actualización:** en la ventana donde corre la plataforma pulse `Ctrl+C`, ejecute de nuevo
`.\iniciar_plataforma.ps1` y, en el navegador, recargue sin caché con `Ctrl+F5`. Los análisis guardados (`platform\data`) se conservan.

Abrir <http://127.0.0.1:8000>. Si ya hay una instancia en marcha, deténgala (Ctrl+C) y vuelva a iniciarla para cargar
la versión nueva. Usa el entorno `unet_oilspill\.venv` (CPU).

**Conexión a internet.** Se usa para: teselas de los mapas (Esri, CARTO) y Leaflet (cdnjs), y para el clima real
(Open-Meteo). A Open-Meteo solo se envían latitud, longitud y fechas del evento. Sin conexión, el análisis y los
informes siguen funcionando; el clima no estará disponible y se podrá simular con viento manual.

## Secciones

| Sección | Qué ofrece |
|---|---|
| Ocean Watch | Escena de mando: mapa a pantalla completa, situación en lenguaje natural, línea de tiempo de eventos y, para el evento elegido, nivel de atención, viento real y costa más cercana |
| Análisis | Resumen sencillo, comparador antes/después, superficie, confianza, ubicación y mapa; bloque «qué podría verse afectado y qué hacer» (clima real, costa, recomendaciones) |
| Mapa | Vista satelital con costas, escala, norte y coordenadas; capas conmutables; detalle y simulación del evento |
| Simulación | **Clima real** (por defecto) o viento manual; origen, trayectoria con marcas horarias, línea a la costa, etapas a 3/6/12 h, reproductor y recomendaciones |
| Historial | Tarjetas con imagen, ubicación (o «desconocida»), resultado, superficie, informe, simular y eliminar |

## Su propia imagen (no solo los ejemplos)

| Situación | Qué hace la plataforma |
|---|---|
| Imagen de **radar** (escala de grises) | Detección automática con el modelo |
| **Foto, captura u otra imagen a color** | No se analiza: se avisa que no parece de radar satelital (Sentinel-1, en blanco y negro) y se pide una imagen de radar |
| La foto trae **GPS** | Se usa como ubicación (la escala sigue siendo aproximada) |
| **Sin ubicación** | Se analiza igual: superficie (con la escala indicada o supuesta, y se avisa) y un **escenario «y si el viento viene de…»** de las 8 direcciones. La simulación se muestra sobre la propia imagen, sin pretender que está en un lugar |
| Ubicación por **nombre** | Búsqueda de lugar (Open-Meteo Geocoding) o coordenadas; se marca como *aproximada* y habilita clima real, costa y mapa. Puede indicarse después de analizar |
| **Escala** | «Ancho aproximado de la imagen (km)» para calcular áreas; si falta, se supone 10 m/px y se declara |

Límites que la plataforma declara: el clima real solo se consulta con ubicación; con ubicación
aproximada, el tamaño de la zona y la distancia a la costa son aproximados.

## Datos reales y datos estimados

| Dato | Fuente | Naturaleza |
|---|---|---|
| Imagen, ubicación | Archivo analizado (GeoTIFF EPSG:4326) | Real; sin georreferencia no se coloca en el mapa |
| Zona marcada | Modelo U-Net | Estimación, no confirma petróleo |
| Viento, olas, corrientes | Open-Meteo | Pronóstico para imágenes recientes; **viento histórico real** de la fecha para los ejemplos de 2020 (olas y corrientes históricas no existen) |
| Costa más cercana | Natural Earth 1:50M (`assets/`) | Real, escala adecuada para decenas de km |
| Desplazamiento | Simulación propia | Ilustrativa; no incluye oleaje, mareas ni química del petróleo |
| Nivel de atención, recomendaciones | Reglas declaradas (`app/context.py`) | Orientativo; no sustituye el plan de contingencia |

Las 173 teselas del dataset están georreferenciadas (Golfo de México, región de Cantarell según su documentación).
Los ejemplos usan esas coordenadas y el satélite/fecha del CSV del dataset (Sentinel-1, 2020).

## Modelo y precisión

- Checkpoint: `unet_oilspill\runs\unet_resnet34_demo_12ep\best.pt` (época 4). No se modifica.
- **Mejora de precisión (sin reentrenar):** el modelo se entrenó con recortes de 256 px y se usa con imágenes completas, y
  las estadísticas de BatchNorm del decoder no se ajustaban a ese tamaño. `tools/recalibrate_bn.py` las recalibra con
  imágenes de **entrenamiento** (sin etiquetas, sin val/test) y guarda `assets/bn_decoder_recal.pt`.
- Punto de operación: umbral 0,95 y descarte de zonas < 100 px.
- Medido en validación con el pipeline entregado: **precisión 0,87** (de lo marcado, ~87 % era derrame de referencia),
  cobertura 0,72, Dice 0,79. Antes: precisión ~0,79, cobertura ~0,76, Dice ~0,77.
- **Cautelas:** el umbral se ajustó en validación (cifras optimistas); cerca de 0,95 hay una meseta, pero por debajo de
  ~0,92 la precisión cae bruscamente. **El conjunto test no se ha evaluado.** Las 43 imágenes de validación contienen
  derrame: no se ha medido cuántas falsas alarmas da en imágenes sin derrame.
- Experimento descartado: entrenar 12 épocas con el encoder congelado (`runs/unet_resnet34_p1_12ep`) no mejoró
  (mejor Dice 0,563 frente a 0,584); se conserva solo como registro.
- Las imágenes con color se marcan con un aviso: el modelo solo conoce radar en escala de grises.

## Si algo falla: diagnóstico

- **Clima (Open-Meteo):** cada fallo se registra con su código HTTP y el motivo real que da el servicio. Se ven en
  `GET /api/status` (campo `weather_recent_errors`) y en el log `oceanguard.weather`. Se distingue entre límite de consultas
  (429, se reintenta con espera), error de petición (400, con el motivo del servicio), error del servidor (5xx), tiempo agotado,
  sin conexión, coordenadas no válidas y fecha no disponible. Con fallo, la plataforma pasa sola al modo **Manual**.
- **Pronóstico, histórico o no disponible:** fecha actual o de hasta 6 días → pronóstico; más antigua → datos históricos reales
  (el fin se recorta a lo disponible); futuro a más de 15 días o anterior a 1940 → no disponible. Sin ubicación confirmada no se
  consulta el clima.
- **Valores no válidos:** la simulación valida entradas y resultados; si algo no es finito, falla con un mensaje claro (nunca se
  sustituye por cero) y la API jamás emite `NaN` (los valores indefinidos van como `null`).
- **Simulación:** la rejilla se adapta al tamaño de la escena (celdas ≥ 100 m, para un paso de tiempo estable) y el dominio se
  dimensiona con el desplazamiento de la mancha, de modo que la trayectoria siempre cabe. En el mapa, «Centrar en el evento»,
  «Ver toda la zona simulada» e «Incluir la costa» reencuadran la vista.

## Pruebas

```powershell
..\unet_oilspill\.venv\Scripts\python.exe -m pytest -q tests
```

Usan un directorio temporal y solo leen el split de validación. Algunas pruebas consultan Open-Meteo y se omiten sin conexión.

## Atribución de los ejemplos incluidos

Los ejemplos de `static/examples/` se redistribuyen con atribución: los de «con derrame» son teselas del *Oil Segmentation Dataset (800x600)* de J.R.M. Garcia (INPE), DOI 10.5281/zenodo.22214144, licencia CC BY 4.0; los de «sin derrame» son recortes de *CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features for machine learning* (Blondeau-Patissier, Schroeder, Diakogiannis y Li), DOI 10.25919/4v55-dn16, licencia CC BY-SA 4.0. Se usan solo para mostrar cómo se ve la pantalla, no para medir precisión.
