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

El sistema implementa una **división exhaustiva de Argentina** en **39 regiones agrícolas especializadas**, optimizadas para análisis completo de estrés hídrico:

#### **🌾 PROVINCIA DE BUENOS AIRES (10 zonas - 0.4° × 0.4°)**
- **Zona 1 (Noroeste)**: [-61.2°, -35.0°] → [-60.8°, -34.6°] - Soja, maíz, trigo
- **Zona 2 (Noreste)**: [-60.8°, -35.0°] → [-60.4°, -34.6°] - Soja, maíz, trigo  
- **Zona 3 (Este)**: [-60.4°, -35.0°] → [-60.0°, -34.6°] - Soja, maíz, trigo
- **Zona 4 (Extremo Este)**: [-60.0°, -35.0°] → [-59.6°, -34.6°] - Soja, maíz, trigo
- **Zona 5 (Centro Oeste)**: [-61.2°, -35.4°] → [-60.8°, -35.0°] - Soja, maíz, trigo
- **Zona 6 (Central)**: [-60.8°, -35.4°] → [-60.4°, -35.0°] - Soja, maíz, trigo
- **Zona 7 (Centro Este)**: [-60.4°, -35.4°] → [-60.0°, -35.0°] - Soja, maíz, trigo
- **Zona 8 (Suroeste)**: [-61.2°, -35.8°] → [-60.8°, -35.4°] - Trigo, girasol, cebada
- **Zona 9 (Centro Sur)**: [-60.8°, -35.8°] → [-60.4°, -35.4°] - Trigo, girasol, cebada
- **Zona 10 (Sureste)**: [-60.4°, -35.8°] → [-60.0°, -35.4°] - Trigo, girasol, cebada

#### **🌽 PROVINCIA DE CÓRDOBA (8 zonas - 0.4° × 0.4°)**
- **Zona 1 (Norte)**: [-64.0°, -31.2°] → [-63.6°, -30.8°] - Soja, maíz, trigo
- **Zona 2 (Noreste)**: [-63.6°, -31.2°] → [-63.2°, -30.8°] - Soja, maíz, trigo
- **Zona 3 (Este)**: [-63.2°, -31.2°] → [-62.8°, -30.8°] - Soja, maíz, trigo
- **Zona 4 (Centro Oeste)**: [-64.0°, -31.6°] → [-63.6°, -31.2°] - Soja, maíz, trigo
- **Zona 5 (Central)**: [-63.6°, -31.6°] → [-63.2°, -31.2°] - Soja, maíz, trigo
- **Zona 6 (Centro Este)**: [-63.2°, -31.6°] → [-62.8°, -31.2°] - Soja, maíz, trigo
- **Zona 7 (Sur)**: [-64.0°, -32.0°] → [-63.6°, -31.6°] - Trigo, girasol, cebada
- **Zona 8 (Sureste)**: [-63.6°, -32.0°] → [-63.2°, -31.6°] - Trigo, girasol, cebada

#### **🌾 PROVINCIA DE SANTA FE (8 zonas - 0.4° × 0.4°)**
- **Zona 1 (Noroeste)**: [-61.4°, -30.0°] → [-61.0°, -29.6°] - Soja, maíz, algodón
- **Zona 2 (Norte)**: [-61.0°, -30.0°] → [-60.6°, -29.6°] - Soja, maíz, algodón
- **Zona 3 (Noreste)**: [-60.6°, -30.4°] → [-60.2°, -30.0°] - Soja, maíz, trigo
- **Zona 4 (Centro Oeste)**: [-61.4°, -30.8°] → [-61.0°, -30.4°] - Soja, maíz, trigo
- **Zona 5 (Central)**: [-61.0°, -30.8°] → [-60.6°, -30.4°] - Soja, maíz, trigo
- **Zona 6 (Centro Este)**: [-60.6°, -30.8°] → [-60.2°, -30.4°] - Soja, maíz, trigo
- **Zona 7 (Sur)**: [-61.4°, -31.2°] → [-61.0°, -30.8°] - Soja, trigo, girasol
- **Zona 8 (Sureste)**: [-61.0°, -31.2°] → [-60.6°, -30.8°] - Soja, trigo, girasol

#### **🌾 ENTRE RÍOS (3 zonas - Mesopotamia)**
- **Zona 1 (Norte)**: [-59.8°, -30.8°] → [-59.4°, -30.4°] - Soja, maíz, arroz
- **Zona 2 (Central)**: [-59.8°, -31.2°] → [-59.4°, -30.8°] - Soja, maíz, arroz
- **Zona 3 (Sur)**: [-59.8°, -31.6°] → [-59.4°, -31.2°] - Soja, trigo, arroz

#### **🌻 LA PAMPA (3 zonas - Agricultura semiárida)**
- **Zona 1 (Este)**: [-65.2°, -36.4°] → [-64.8°, -36.0°] - Trigo, girasol, maíz
- **Zona 2 (Central)**: [-65.6°, -36.4°] → [-65.2°, -36.0°] - Trigo, girasol, cebada
- **Zona 3 (Norte)**: [-65.2°, -36.0°] → [-64.8°, -35.6°] - Trigo, girasol, maíz

#### **🌾 REGIONES NORTEÑAS (5 zonas - Agricultura subtropical)**
- **Santiago del Estero Zona 1**: [-63.4°, -28.0°] → [-63.0°, -27.6°] - Soja, algodón, trigo
- **Santiago del Estero Zona 2**: [-63.0°, -28.0°] → [-62.6°, -27.6°] - Soja, algodón, trigo
- **Chaco Zona 1 (Sur)**: [-60.8°, -27.0°] → [-60.4°, -26.6°] - Algodón, soja, girasol
- **Chaco Zona 2 (Central)**: [-60.4°, -27.0°] → [-60.0°, -26.6°] - Algodón, soja, girasol
- **Tucumán Zona 1**: [-65.6°, -26.8°] → [-65.2°, -26.4°] - Caña de azúcar, soja, cítricos
- **Salta Zona 1**: [-65.2°, -25.2°] → [-64.8°, -24.8°] - Soja, caña de azúcar, porotos

#### **📊 ZONA DE PRUEBA (1 región)**
- **Área Micro Test**: [-60.0°, -34.0°] → [-59.8°, -33.8°] - Testing y validación

### **🎯 Características del Sistema Regional**

- **Cobertura Total**: 39 regiones especializadas (~1,600 km² cada una)
- **Resolución Espacial**: 0.4° × 0.4° para manejo eficiente de datos
- **Priorización**: Alta (26 regiones Pampa), Media (10 regiones), Baja (2 regiones), Test (1 región)
- **Análisis Simultáneo**: Procesamiento paralelo de todas las regiones
- **Tiempo de Análisis**: ~5-6 segundos por región (total: 3-4 minutos)
- **Cobertura Agrícola**: >95% de la superficie agrícola argentina

## Uso

### Uso básico

#### Análisis completo de Argentina (39 regiones)
Ejecutar el análisis completo de todas las regiones agrícolas por defecto:
```bash
python main.py
```
**Resultado**: Análisis automático de las 39 regiones en 3-4 minutos, generando:
- Reportes individuales por región
- Mapa interactivo nacional consolidado  
- Resumen nacional de alertas
- Distribución de estrés hídrico por provincia

### Uso avanzado

#### Análisis de región específica
```bash
# Analizar una región específica solamente
python main.py --single-region --study-area buenos_aires_01_northwest --crop-type soybean

# Analizar solo regiones prioritarias (4 regiones clave)
python main.py --study-area buenos_aires_01_northwest --crop-type soybean

# Especificar rango de fechas para análisis nacional
python main.py --start-date 2024-01-01 --end-date 2024-03-31

# Usar configuración personalizada
python main.py --config custom_config.json
```

#### Regiones disponibles para análisis individual:
```bash
# Buenos Aires (10 zonas)
--study-area buenos_aires_01_northwest
--study-area buenos_aires_02_northeast
# ... hasta buenos_aires_10_southeast

# Córdoba (8 zonas)  
--study-area cordoba_01_north
--study-area cordoba_02_northeast
# ... hasta cordoba_08_southeast

# Santa Fe (8 zonas)
--study-area santa_fe_01_northwest
--study-area santa_fe_02_north
# ... hasta santa_fe_08_southeast

# Entre Ríos (3 zonas)
--study-area entre_rios_01_north
--study-area entre_rios_02_central
--study-area entre_rios_03_south

# La Pampa (3 zonas)
--study-area la_pampa_01_east
--study-area la_pampa_02_central  
--study-area la_pampa_03_north

# Regiones Norteñas (5 zonas)
--study-area santiago_estero_01
--study-area chaco_01_south
--study-area tucuman_01
# ... etc
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
