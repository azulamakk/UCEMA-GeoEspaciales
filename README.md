# Sistema de Detección Temprana de Estrés Hídrico en Cultivos Extensivos

Un sistema integral para detectar y predecir el estrés hídrico en cultivos extensivos (soja, maíz, trigo) en Argentina utilizando APIs de código abierto y aprendizaje automático.

## Características

- **Integración de datos multi-fuente**: Imágenes satelitales Sentinel-2, datos meteorológicos NASA POWER, propiedades del suelo de SoilGrids
- **Índices de vegetación avanzados**: NDVI, GNDVI, NDWI, CWSI, EVI, SAVI, LAI
- **Predicciones con aprendizaje automático**: Modelos Random Forest y XGBoost para pronóstico de estrés hídrico
- **Sistema de alertas tempranas**: Alertas automatizadas con recomendaciones accionables
- **Mapas de prescripción**: Recomendaciones de riego variable
- **Minimización de datos**: Cumplimiento con privacidad de datos y optimización de almacenamiento
- **Análisis de series temporales**: Detección de tendencias, identificación de anomalías, descomposición estacional

## Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Datos Satelitales│    │ Datos Meteorológicos│   │   Datos Suelo    │
│   (Sentinel-2)  │    │  (NASA POWER)   │    │  (SoilGrids)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │  Procesamiento de Datos & │
                    │ Ingeniería de Características│
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │ Análisis Series Temporales│
                    │  & Detección Anomalías    │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │  Aprendizaje Automático   │
                    │   (RF + XGBoost)         │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
│ Sistema Alertas │    │   Mapas de      │    │ Gestión de Datos│
│                │    │ Prescripción    │    │  & Calidad      │
└────────────────┘    └─────────────────┘    └─────────────────┘
```

## Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd water_stress_detection
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Configurar autenticación de Google Earth Engine**:
```bash
earthengine authenticate
```

4. **Configurar variables de entorno**:
```bash
cp config/.env.example .env
# Editar .env con tus credenciales de API
```

## Configuración

### Configuración de Google Earth Engine

1. Crear una cuenta en Google Earth Engine en https://earthengine.google.com/
2. Crear una cuenta de servicio y descargar el archivo JSON de la clave
3. Establecer la ruta en tu archivo .env:
```
GEE_SERVICE_ACCOUNT_KEY=/path/to/your/service-account-key.json
GEE_PROJECT_ID=your-gee-project-id
```

### Áreas de Estudio

El sistema viene preconfigurado con regiones agrícolas de Argentina:
- **Región Pampeana**: Principal área agrícola
- **Provincia de Córdoba**: Producción intensiva de cultivos
- **Zona Agrícola de Buenos Aires**: Sistemas de cultivos diversos

## Uso

### Uso básico
Ejecutar el análisis completo para el área de estudio por defecto:
```bash
python main.py
```

### Uso avanzado

```bash
# Analizar un área de estudio y cultivo específicos
python main.py --study-area pampas_region --crop-type soybean

# Especificar rango de fechas
python main.py --start-date 2024-01-01 --end-date 2024-03-31

# Usar configuración personalizada
python main.py --config custom_config.json
```

### Uso programático

```python
from water_stress_detection.main import WaterStressDetectionSystem
import asyncio

# Inicializar sistema
system = WaterStressDetectionSystem()

# Ejecutar análisis
results = asyncio.run(system.run_full_analysis(
    study_area='pampas_region',
    crop_type='soybean',
    start_date='2024-01-01',
    end_date='2024-03-31'
))

# Acceder a resultados
alert = results['alerts']['current_alert']
print(f"Nivel de alerta: {alert['alert_level']}")

```

## Fuentes de datos

### Datos satelitales (Google Earth Engine)
- **Sentinel-2**: Resolución de 10 m, tiempo de revisita de 5 días
- **Landsat 8/9**: Datos térmicos para cálculo de CWSI
- **Enmascaramiento de nubes**: Filtrado de calidad automatizado

### Datos meteorológicos (NASA POWER API)
- **Variables**: Temperatura, precipitación, evapotranspiración, déficit de presión de vapor
- **Resolución**: Diaria, 0.5° x 0.625°
- **Cobertura**: Global, desde 1981 hasta la actualidad

### Datos de suelo (SoilGrids API)
- **Propiedades**: Textura, densidad aparente, carbono orgánico, pH
- **Resolución**: 250m
- **Profundidades**: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm

## Índices de vegetación

| Índice | Fórmula | Propósito |
|--------|---------|-----------|
| NDVI | (NIR - Rojo) / (NIR + Rojo) | Salud de la vegetación |
| GNDVI | (NIR - Verde) / (NIR + Verde) | Contenido de clorofila |
| NDWI | (Verde - NIR) / (Verde + NIR) | Contenido de agua en plantas |
| CWSI | Basado en temperatura | Estrés hídrico de cultivos |
| EVI | Índice de vegetación mejorado | Mayor sensibilidad |
| SAVI | Índice ajustado por suelo | Reducción de influencia del suelo |

## Modelos de Aprendizaje Automático

### Clasificador Random Forest
- **Características**: +50 características ingenierizadas
- **Objetivo**: Nivel de estrés hídrico (escala 0-2)
- **Validación**: Validación cruzada de series temporales
- **Rendimiento**: Típicamente >85% de precisión

### Clasificador XGBoost
- **Características**: El mismo conjunto de características que Random Forest
- **Hiperparámetros**: Optimización con búsqueda en grilla
- **Parada temprana**: Previene el sobreajuste
- **Importancia de características**: Valores SHAP disponibles

## Sistema de Alertas

### Niveles de Alerta
- **Normal**: No se detecta estrés
- **Advertencia**: Indicadores tempranos de estrés
- **Crítico**: Acción inmediata requerida

### Indicadores
- Índices de vegetación por debajo de los umbrales
- Balance hídrico negativo
- Temperaturas altas
- Días consecutivos secos
- Anomalías estadísticas

### Recomendaciones
- Programación de riego
- Estrategias de mitigación de estrés
- Ajustes en frecuencia de monitoreo
- Consejos específicos por cultivo

## Mapas de Prescripción

### Riego de Tasa Variable
- **Zonas**: 5 zonas de manejo de riego
- **Tasas**: 0-20 mm/día basadas en el nivel de estrés
- **Eficiencia**: Considera la eficiencia del sistema de riego
- **Formatos**: GeoTIFF, mapas interactivos, JSON

### Integración con Equipos
- Parámetros de ancho de barra y velocidad
- Recomendaciones de selección de boquillas
- Cálculos de presión y caudal
- Estimaciones de tiempo de aplicación

## Gestión de Datos

### Minimización de Datos
- **Datos esenciales**: Modelos, alertas, prescripciones (almacenamiento permanente)
- **Datos temporales**: Datos satelitales/meteorológicos crudos (retención 30 días)
- **Limpieza automática**: Eliminación programada de datos vencidos
- **Cumplimiento de privacidad**: Opciones de anonimización de ubicación

### Control de Calidad
- **Validación**: Verificación de rangos, detección de valores atípicos, completitud
- **Métricas**: Puntuaciones de calidad, banderas de anomalías, niveles de confianza
- **Reportes**: Informes de calidad automatizados con recomendaciones

## Metodología de Validación

### Integración de Datos de Campo
- **Sensores de humedad del suelo**: Datos de validación opcionales
- **Modelos de cultivos**: Comparaciones con modelos de código abierto (DSSAT, APSIM)
- **Observaciones de campo**: Integración con instituciones agrícolas
- **Correlación de rendimiento**: Validación con datos históricos de rendimiento

### Métricas de Rendimiento
- **Precisión**: Precisión de clasificación para detección de estrés
- **Precisión/Sensibilidad**: Métricas de rendimiento por clase
- **Puntuación F1**: Medida de rendimiento equilibrada
- **AUC-ROC**: Discriminación general del modelo

## Límites de APIs y Optimización

### Google Earth Engine
- **Cuota**: Varía según el tipo de cuenta
- **Optimización**: Filtrado espacial/temporal eficiente
- **Procesamiento por lotes**: Múltiples operaciones por solicitud

### NASA POWER
- **Límite de tasa**: Sin límite explícito
- **Optimización**: Solicitudes asíncronas para múltiples ubicaciones
- **Caché**: Almacenamiento local de datos históricos

### SoilGrids
- **Límite de tasa**: Sin límite explícito
- **Optimización**: Solicitudes de coordenadas por lotes
- **Caché**: Almacenamiento persistente de datos de suelo

## Solución de Problemas

### Problemas Comunes

1. **Autenticación de Google Earth Engine**:
```bash
earthengine authenticate --force
```

2. **Dependencias Faltantes**:
```bash
pip install --upgrade -r requirements.txt
```

3. **Problemas de Memoria con Áreas Grandes**:
   - Reducir período de análisis
   - Aumentar parámetro de resolución de grilla
   - Usar submuestreo espacial

4. **Tiempos de Espera de API**:
   - Verificar conexión a internet
   - Reducir tamaño de solicitud
   - Implementar lógica de reintento

### Códigos de Error

- **EE001**: Falló la autenticación de Earth Engine
- **API002**: Tiempo de espera agotado de API de datos meteorológicos
- **ML003**: Datos de entrenamiento insuficientes
- **VAL004**: Falló la validación de calidad de datos
