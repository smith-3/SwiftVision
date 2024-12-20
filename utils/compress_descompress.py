# compress_descompress.py

import cv2
import numpy as np
import json
from typing import List, Tuple
import logging

# Configurar el logger
logger = logging.getLogger(__name__)

def combine_boolean_matrices(matrices):
    """
    Combina múltiples matrices booleanas usando la operación lógica OR.

    :param matrices: Lista de matrices booleanas numpy.
    :return: Una sola matriz booleana combinada o None en caso de error.
    """
    try:
        if not matrices:
            raise ValueError("La lista de matrices está vacía.")

        # Verificar que todas las matrices tengan el mismo tamaño
        shape = matrices[0].shape
        logger.debug(f"Tamaño de la primera matriz: {shape}")  # Usar logger en lugar de print
        for i, matrix in enumerate(matrices[1:], start=1):
            logger.debug(f"Tamaño de la matriz {i}: {matrix.shape}")  # Usar logger en lugar de print
            if matrix.shape != shape:
                raise ValueError(f"Todas las matrices deben tener el mismo tamaño. Error en la matriz {i}: tamaño esperado {shape}, tamaño encontrado {matrix.shape}")

        # Combinar las matrices usando np.logical_or
        combined_matrix = np.zeros(shape, dtype=bool)
        for matrix in matrices:
            combined_matrix = np.logical_or(combined_matrix, matrix)

        return combined_matrix

    except ValueError as e:
        logger.error(f"Error en combine_boolean_matrices: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado en combine_boolean_matrices: {e}")
        return None  # También puedes lanzar la excepción nuevamente si prefieres

def getMask(mask: np.ndarray) -> bool:
    # Obtener las dimensiones de la matriz
    height, width = mask.shape[:2]

    # Calcular las posiciones como enteros
    posX = int(width / 2)
    posY = int(height / 3)

    # Acceder al valor en la posición calculada
    return bool(mask[posY, posX])

def convert_to_json(compressed_counts):
    return json.dumps(compressed_counts)

def convert_from_json(compressed_counts_json):
    return json.loads(compressed_counts_json)

def load_mask(mask_info: dict):
    counts = np.array(mask_info['segmentation']['counts'])
    size = tuple(mask_info['segmentation']['size'])
    segmentation_array = counts.reshape(size).astype(bool)

    return {
        "segmentation": segmentation_array,
        "bbox": mask_info['bbox'],
        "area": mask_info['area'],
        "predicted_iou": mask_info['predicted_iou'],
        "stability_score": mask_info['stability_score'],
        "crop_box": mask_info['crop_box'],
        "point_coords": mask_info['point_coords']
    }

def run_length_encode_row(row: np.ndarray) -> List[Tuple[int, int]]:
    changes = np.diff(row) != 0
    idx = np.where(changes)[0] + 1
    counts = np.diff(np.concatenate(([0], idx, [len(row)])))
    values = row[np.concatenate(([0], idx))].astype(int)
    return [(int(count), int(value)) for count, value in zip(counts, values)]

def run_length_encode_matrix(matrix: np.ndarray) -> List[List[Tuple[int, int]]]:
    return [run_length_encode_row(row) for row in matrix]

def compress_encoded_matrix(encoded_matrix: List[List[Tuple[int, int]]]) -> List[Tuple[List[Tuple[int, int]], int]]:
    compressed_counts = []
    if not encoded_matrix:
        return compressed_counts

    current_row = encoded_matrix[0]
    current_count = 1

    for row in encoded_matrix[1:]:
        if row == current_row:
            current_count += 1
        else:
            compressed_counts.append((current_row, current_count))
            current_row = row
            current_count = 1

    compressed_counts.append((current_row, current_count))
    return compressed_counts

def decompress_encoded_matrix(compressed_counts: List[Tuple[List[Tuple[int, int]], int]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Descomprime la matriz desde el formato comprimido a una matriz booleana.
    """
    decoded_matrix = np.zeros(shape, dtype=bool)
    y = 0  # Índice de la fila que se está llenando en la matriz final

    for row_tuple, count in compressed_counts:
        # Reconstruir una fila decodificada basada en las tuplas (longitud, valor)
        decoded_row = np.array([value for length, value in row_tuple for _ in range(length)], dtype=bool)

        # Log de depuración
        logger.debug(f"Decoding row {y}: expected width={shape[1]}, decoded_row length={decoded_row.shape[0]}")

        # Validar que el ancho de la fila coincida con el esperado
        if decoded_row.shape[0] != shape[1]:
            raise ValueError(
                f"La fila decodificada tiene un ancho inesperado. "
                f"Esperado: {shape[1]}, Obtenido: {decoded_row.shape[0]}"
            )

        # Repetir la fila 'count' veces y añadirla a la matriz final
        for _ in range(count):
            if y >= shape[0]:
                raise ValueError(
                    "Se ha excedido el número esperado de filas en la matriz decodificada."
                )
            decoded_matrix[y] = decoded_row
            y += 1

    # Validar que se llenaron todas las filas esperadas
    if y != shape[0]:
        raise ValueError(
            f"El número de filas decodificadas no coincide con el esperado. "
            f"Esperado: {shape[0]}, Obtenido: {y}"
        )

    logger.info(f"Descompresión completada con éxito. Shape final: {decoded_matrix.shape}")
    return decoded_matrix


def compress_image_bytes(image_bytes: bytes, max_side: int = 1200) -> bytes:
    # Convertir los bytes a una imagen
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR para cargar en BGR
    
    if image is None:
        raise ValueError("No se pudo decodificar la imagen desde los bytes.")
    
    # Obtener las dimensiones originales
    height, width = image.shape[:2]
    print(f"Tamaño original de la imagen: {width}x{height}")
    
    # Verificar si es necesario redimensionar
    if max(height, width) > max_side:
        # Calcular el factor de escalado
        scaling_factor = max_side / max(height, width)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        
        # Redimensionar la imagen
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Tamaño de la imagen después de redimensionar: {new_width}x{new_height}")
    else:
        print("La imagen no requiere redimensionamiento.")
    
    # Convertir la imagen comprimida de nuevo a bytes
    # Usamos cv2.imencode para codificar la imagen de vuelta en bytes
    _, image_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])  # 90 es el nivel de calidad

    # Convertir a bytes y devolver
    compressed_image_bytes = image_encoded.tobytes()
    
    return compressed_image_bytes


if __name__ == "__main__":
    counts = np.array([
        [False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, True],
        [True, True, True, True, False, False, False, False, False, True],
        [False, False, False, False, True, True, True, True, True, False],
        [False, False, False, True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [True, True, True, True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
    ], dtype=bool)

    encoded_matrix = run_length_encode_matrix(counts)
    compressed_counts = compress_encoded_matrix(encoded_matrix)

    # Convertir a JSON
    compressed_counts_json = convert_to_json(compressed_counts)
    print("Compressed Counts (JSON):")
    print(compressed_counts_json)

    # Convertir de JSON
    compressed_counts_from_json = convert_from_json(compressed_counts_json)

    print(str(counts.shape))
    # Descomprime la matriz
    decoded_matrix = decompress_encoded_matrix(compressed_counts_from_json, counts.shape)
    print(decoded_matrix)
