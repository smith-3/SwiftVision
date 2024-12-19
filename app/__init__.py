import logging

# Configura el logger global
logging.basicConfig(
    level=logging.INFO,  # Cambia a WARNING o ERROR para reducir la verbosidad
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configura el logger para librerías específicas si es necesario
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("modelsAI.lama").setLevel(logging.WARNING)
