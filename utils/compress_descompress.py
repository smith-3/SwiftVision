import numpy as np
import json
from collections import defaultdict

def run_length_encode_row(row):
    changes = np.diff(row) != 0
    idx = np.where(changes)[0] + 1
    counts = np.diff(np.concatenate(([0], idx, [len(row)])))
    values = row[np.concatenate(([0], idx))].astype(int)
    # Convertir a enteros nativos de Python
    return [(int(count), int(value)) for count, value in zip(counts, values)]

def run_length_encode_matrix(matrix):
    return [run_length_encode_row(row) for row in matrix]

def compress_encoded_matrix(encoded_matrix):
    compressed_counts = []
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

def decompress_encoded_matrix(compressed_counts, shape):

    decoded_matrix = np.zeros(shape, dtype=bool)
    y = 0  # Índice de la fila que se está llenando en la matriz final


    for index, (row_tuple, count) in enumerate(compressed_counts):
     

        # Reconstruir una fila decodificada basada en las tuplas (longitud, valor)
        decoded_row = []
        for length, value in row_tuple:
            decoded_row.extend([value] * length)

        # Convertir a numpy array
        row_np = np.array(decoded_row, dtype=bool)
        

        # Validar que el ancho de la fila coincida con el de la matriz esperada
        if row_np.shape[0] != shape[1]:
            raise ValueError(f"La fila decodificada tiene un ancho inesperado. Esperado: {shape[1]}, Obtenido: {row_np.shape[0]}")

        # Repetir la fila 'count' veces y añadirla a la matriz final
        for repeat in range(count):
            if y >= shape[0]:
                raise ValueError("Se ha excedido el número esperado de filas en la matriz decodificada.")
            decoded_matrix[y] = row_np
            
            y += 1

    # Validar que se llenaron todas las filas esperadas
    if y != shape[0]:
        raise ValueError(f"El número de filas decodificadas no coincide con el esperado. Esperado: {shape[0]}, Obtenido: {y}")

    return decoded_matrix


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
        print(f"Tamaño de la primera matriz: {shape}")  # Imprimir tamaño de la primera matriz
        for i, matrix in enumerate(matrices[1:], start=1):
            print(f"Tamaño de la matriz {i}: {matrix.shape}")  # Imprimir tamaño de cada matriz
            if matrix.shape != shape:
                raise ValueError(f"Todas las matrices deben tener el mismo tamaño. Error en la matriz {i}: tamaño esperado {shape}, tamaño encontrado {matrix.shape}")

        # Combinar las matrices usando np.logical_or
        combined_matrix = np.zeros(shape, dtype=bool)
        for matrix in matrices:
            combined_matrix = np.logical_or(combined_matrix, matrix)

        return combined_matrix
    
    except Exception as e:
        print(f"Error en combine_boolean_matrices: {e}")
        return None

    except ValueError as e:
        print(f"Error en combine_boolean_matrices: {e}")
        return None  # Puedes retornar None o manejar el error de otra forma
    except Exception as e:
        print(f"Error inesperado en combine_boolean_matrices: {e}")
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
    
    return{
        "segmentation": segmentation_array,
        "bbox": mask_info['bbox'],
        "area": mask_info['area'],
        "predicted_iou": mask_info['predicted_iou'],
        "stability_score": mask_info['stability_score'],
        "crop_box": mask_info['crop_box'],
        "point_coords": mask_info['point_coords']
    }
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
