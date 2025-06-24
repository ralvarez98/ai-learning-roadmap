import requests
import time

# URL del maestro IO-Link
url = "http://192.168.0.248/iolinkmaster/port[2]/iolinkdevice/pdin/getdata"

def hex_to_ascii(hex_str):
    """Convierte una cadena hexadecimal en texto ASCII."""
    try:
        bytes_data = bytes.fromhex(hex_str)
        return bytes_data.decode('ascii', errors='ignore')  # Ignorar caracteres no ASCII
    except ValueError as e:
        print(f"Error en la conversión de hexadecimal a ASCII: {e}")
        return None

def fetch_qr_data():
    """Recibe y organiza los segmentos en el orden correcto."""
    retries = 10
    attempt = 0
    received_segments = {}  # Diccionario para almacenar segmentos por su Segment Counter

    while attempt < retries:
        try:
            print(f"Intento {attempt + 1} de {retries}...")
            response = requests.get(url)
            response.raise_for_status()

            # Imprimir datos crudos
            print(f"Respuesta completa del dispositivo: {response.text}")

            # Procesar datos recibidos
            data = response.json()
            if 'data' in data and 'value' in data['data']:
                hex_value = data['data']['value']
                print(f"32 bytes recibidos (hex): {hex_value}")

                # Extraer encabezado y datos útiles
                segment_counter = int(hex_value[4:6], 16)  # Byte 3: Segment Counter
                input_data = hex_value[8:]  # Datos útiles (últimos 28 bytes)
                ascii_data = hex_to_ascii(input_data)

                print(f"Segment Counter: {segment_counter}")
                print(f"Datos ASCII del segmento: {ascii_data}")

                # Almacenar segmento por su número
                if segment_counter not in received_segments:
                    received_segments[segment_counter] = ascii_data
                else:
                    print(f"Segmento {segment_counter} ya recibido.")

                # Reconstruir mensaje si sabemos que todos los segmentos están presentes
                if len(received_segments) == 2:  # Esperamos exactamente 2 segmentos
                    # Concatenar en el orden correcto: primero el segmento 1 (URL) y luego el segmento 0 (QR Code)
                    complete_qr_data = received_segments.get(1, '') + received_segments.get(0, '')
                    print(f"Contenido completo del QR (ordenado): {complete_qr_data}")
                    return
            else:
                print("No se encontraron datos válidos en la respuesta.")
        except requests.exceptions.RequestException as e:
            print(f"Error de comunicación: {e}")
            break

        attempt += 1
        time.sleep(1)

    if len(received_segments) == 0:
        print("No se recibieron datos válidos. Intenta nuevamente.")
    else:
        print("Datos incompletos, asegúrate de que el sensor esté configurado correctamente.")

if __name__ == "__main__":
    print("Iniciando recepción desde el dispositivo IO-Link...")
    fetch_qr_data()
