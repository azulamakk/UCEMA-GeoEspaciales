# Sistema de Detección de Estrés Hídrico - Instrucciones de Configuración

## ✅ Pasos de Configuración Completados

1. ✅ **Dependencias instaladas** - Todos los paquetes de Python de requirements.txt
2. ✅ **Directorios creados** - Directorios `outputs/` y `models/`
3. ✅ **Archivo de entorno creado** - Archivo plantilla `.env`
4. ✅ **Autenticación de Earth Engine** - Token OAuth guardado
5. ✅ **Conectividad de APIs probada** - APIs de NASA POWER y SoilGrids funcionando

## ⚠️ Pasos Manuales Requeridos

### 1. Configuración del Proyecto de Google Earth Engine

**IMPORTANTE**: Necesitas configurar un proyecto de Google Earth Engine:

1. **Ve a Google Earth Engine**: https://earthengine.google.com/
2. **Crea o selecciona un proyecto**:
   - Si no tienes uno, crea un nuevo proyecto
   - Anota tu ID de proyecto (ej., "mi-proyecto-gee-12345")

3. **Actualiza el archivo `.env`** con tu ID de proyecto real:
   ```bash
   # Edita este archivo: /Users/azulmakk/Desktop/Datos geoespaciales/water_stress_detection/.env
   GEE_PROJECT_ID=tu-id-de-proyecto-actual-aqui
   ```

### 2. Alternativa: Configuración de Cuenta de Servicio (Opcional pero Recomendado)

Para uso en producción, considera configurar una cuenta de servicio:

1. **Consola de Google Cloud**: https://console.cloud.google.com/
2. **Crear cuenta de servicio** en tu proyecto GEE
3. **Descargar archivo de clave JSON**
4. **Actualizar archivo `.env`**:
   ```bash
   GEE_SERVICE_ACCOUNT_KEY=/ruta/a/tu/archivo-clave-cuenta-servicio.json
   GEE_PROJECT_ID=tu-id-proyecto-gee
   GEE_USE_SERVICE_ACCOUNT=true
   ```

## 🧪 Probar el Sistema

Después de actualizar tu archivo `.env` con el ID de proyecto correcto:

```bash
cd "/Users/azulmakk/Desktop/Datos geoespaciales/water_stress_detection"
python test_apis.py
```

## 🚀 Ejecutar el Sistema

Una vez que todas las pruebas pasen, puedes ejecutar el análisis completo:

```bash
# Ejecución básica con parámetros por defecto
python main.py

# Ejecutar con parámetros específicos
python main.py --study-area pampas_region --crop-type soybean --start-date 2024-01-01 --end-date 2024-03-31
```

## 📁 Estructura Actual del Proyecto

```
water_stress_detection/
├── .env                     # ✅ Variables de entorno (necesita GEE_PROJECT_ID)
├── requirements.txt         # ✅ Dependencias instaladas
├── main.py                  # Punto de entrada principal del sistema
├── test_apis.py            # ✅ Prueba de conectividad de APIs
├── outputs/                # ✅ Creado para resultados de análisis
├── models/                 # ✅ Creado para modelos de ML
├── config/
│   ├── api_config.py
│   └── crop_parameters.py
├── data_acquisition/
│   ├── satellite_data.py
│   ├── weather_data.py
│   └── soil_data.py
└── [otros módulos...]
```

## 🔧 Solución de Problemas

### Problemas Comunes:

1. **"no project found"** → Actualiza GEE_PROJECT_ID en el archivo .env
2. **"Authentication failed"** → Ejecuta `earthengine authenticate` nuevamente
3. **"Module not found"** → Asegúrate de estar en el directorio correcto y entorno virtual

### Recursos de Soporte:

- **Guía de Google Earth Engine**: https://developers.google.com/earth-engine/guides/python_install
- **Configuración de Proyecto GEE**: https://developers.google.com/earth-engine/guides/service_account
- **Documentación del Sistema**: Ver README.md para instrucciones detalladas de uso