import os

# Directorio fijo donde se crearán las carpetas (modifica esta ruta)
DIRECTORIO_BASE = "C:\\Users\\Coshita\\Downloads\\stroke datos de entrenamiento\\DWI\\DWI Clase Stroke Imagenes"  # Para Windows


# DIRECTORIO_BASE = "/home/usuario/pacientes"  # Para Linux/Mac

def crear_carpetas_pacientes():
    # Verificar si el directorio existe, si no, crearlo
    if not os.path.exists(DIRECTORIO_BASE):
        try:
            os.makedirs(DIRECTORIO_BASE)
            print(f"Directorio base creado: {DIRECTORIO_BASE}")
        except Exception as e:
            print(f"No se pudo crear el directorio base: {e}")
            return

    # Crear las 20 carpetas (paciente1 a paciente14)
    for i in range(1, 15):
        nombre_carpeta = f"paciente{i}"
        ruta_completa = os.path.join(DIRECTORIO_BASE, nombre_carpeta)

        try:
            os.mkdir(ruta_completa)
            print(f"Carpeta creada: {ruta_completa}")
        except FileExistsError:
            print(f"La carpeta {ruta_completa} ya existe - omitiendo")
        except Exception as e:
            print(f"Error al crear {ruta_completa}: {e}")

    print("\nProceso completado. Carpetas creadas desde paciente1 hasta paciente14")


# Ejecutar la función
crear_carpetas_pacientes()