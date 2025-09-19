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

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Satellite Data │    │  Weather Data   │    │   Soil Data     │
│   (Sentinel-2)  │    │  (NASA POWER)   │    │  (SoilGrids)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Data Processing &      │
                    │   Feature Engineering     │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Time Series Analysis    │
                    │   & Anomaly Detection     │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Machine Learning        │
                    │   (RF + XGBoost)          │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
│  Alert System  │    │ Prescription    │    │ Data Management │
│                │    │     Maps        │    │ & Quality       │
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
# Edit .env with your API credentials
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

### Study Areas

El sistema viene preconfigurado con regiones agrícolas de Argentina:
- **Región Pampeana**: Principal área agrícola
- **Provincia de Córdoba**: Producción intensiva de cultivos
- **Zona Agrícola de Buenos Aires**: Sistemas de cultivos diversos

## Uso

### Uso basico
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

### Uso programatico

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

### Satellite Data (Google Earth Engine)
- **Sentinel-2**: Resolución de 10 m, tiempo de revisita de 5 días
- **Landsat 8/9**: Datos térmicos para cálculo de CWSI
- **Enmascaramiento de nubes**: Filtrado de calidad automatizado

### Datos meteorológicos (NASA POWER API)
- **Variables**: Temperatura, precipitación, evapotranspiración, déficit de presión de vapor
- **Resolution**: Diaria, 0.5° x 0.625°
- **Coverage**: Global, desde 1981 hasta la actualidad

### Datos de suelo (SoilGrids API)
- **Properties**: Textura, densidad aparente, carbono orgánico, pH
- **Resolution**: 250m
- **Depths**: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm

## Indices de vegetación

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (NIR - Red) / (NIR + Red) | Salud de la vegetación |
| GNDVI | (NIR - Green) / (NIR + Green) | Contenido de clorofila |
| NDWI | (Green - NIR) / (Green + NIR) | Contenido de agua en plantas |
| CWSI | Basado en temperatura | Estrés hídrico de cultivos |
| EVI | Índice de vegetación mejorado | Mayor sensibilidad |
| SAVI | Indice ajustado por suelo | Reducción de influencia del suelo |

## Machine Learning Models

### Random Forest Classifier
- **Features**: 50+ engineered features
- **Target**: Water stress level (0-2 scale)
- **Validation**: Time series cross-validation
- **Performance**: Typically >85% accuracy

### XGBoost Classifier
- **Features**: Same feature set as Random Forest
- **Hyperparameters**: Grid search optimization
- **Early stopping**: Prevents overfitting
- **Feature importance**: SHAP values available

## Alert System

### Alert Levels
- **Normal**: No stress detected
- **Warning**: Early stress indicators
- **Critical**: Immediate action required

### Indicators
- Vegetation indices below thresholds
- Negative water balance
- High temperatures
- Consecutive dry days
- Statistical anomalies

### Recommendations
- Irrigation scheduling
- Stress mitigation strategies
- Monitoring frequency adjustments
- Crop-specific advice

## Prescription Maps

### Variable Rate Irrigation
- **Zones**: 5 irrigation management zones
- **Rates**: 0-20 mm/day based on stress level
- **Efficiency**: Accounts for irrigation system efficiency
- **Formats**: GeoTIFF, interactive maps, JSON

### Equipment Integration
- Boom width and speed parameters
- Nozzle selection recommendations
- Pressure and flow rate calculations
- Application time estimates

## Data Management

### Data Minimization
- **Essential data**: Models, alerts, prescriptions (permanent storage)
- **Temporary data**: Raw satellite/weather data (30-day retention)
- **Automatic cleanup**: Scheduled removal of expired data
- **Privacy compliance**: Location anonymization options

### Quality Control
- **Validation**: Range checks, outlier detection, completeness
- **Metrics**: Quality scores, anomaly flags, confidence levels
- **Reporting**: Automated quality reports with recommendations

## Validation Methodology

### Ground Truth Integration
- **Soil moisture sensors**: Optional validation data
- **Crop models**: Open-source model comparisons (DSSAT, APSIM)
- **Field observations**: Integration with agricultural institutions
- **Yield correlation**: Historical yield data validation

### Performance Metrics
- **Accuracy**: Classification accuracy for stress detection
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Overall model discrimination

## API Rate Limits and Optimization

### Google Earth Engine
- **Quota**: Varies by account type
- **Optimization**: Efficient spatial/temporal filtering
- **Batching**: Multiple operations per request

### NASA POWER
- **Rate limit**: No explicit limit
- **Optimization**: Asynchronous requests for multiple locations
- **Caching**: Local storage of historical data

### SoilGrids
- **Rate limit**: No explicit limit
- **Optimization**: Bulk coordinate requests
- **Caching**: Persistent soil data storage

## Troubleshooting

### Common Issues

1. **Google Earth Engine Authentication**:
```bash
earthengine authenticate --force
```

2. **Missing Dependencies**:
```bash
pip install --upgrade -r requirements.txt
```

3. **Memory Issues with Large Areas**:
   - Reduce analysis period
   - Increase grid resolution parameter
   - Use spatial subsampling

4. **API Timeouts**:
   - Check internet connection
   - Reduce request size
   - Implement retry logic

### Error Codes

- **EE001**: Earth Engine authentication failed
- **API002**: Weather data API timeout
- **ML003**: Insufficient training data
- **VAL004**: Data quality validation failed

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request


## Agradecimientos
Google Earth Engine por el acceso a datos satelitales
NASA POWER por los datos meteorológicos
ISRIC SoilGrids por los datos de propiedades de suelo
Instituciones agrícolas argentinas por la experiencia en el dominio
Comunidad open-source por las herramientas y librerías
