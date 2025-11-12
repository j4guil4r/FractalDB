import os
from PIL import Image, ImageDraw, ImageFont

# Directorio donde se guardarán (el mismo que montas en Docker)
output_dir = "data_imgs"
os.makedirs(output_dir, exist_ok=True)

print(f"Creando 15 imágenes de prueba en '{output_dir}'...")

# 5 cuadrados rojos (simulando "Carros")
for i in range(5):
    filename = os.path.join(output_dir, f"carro{i+1}.jpg")
    img = Image.new('RGB', (200, 200), color = 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle(((25, 25), (175, 175)), fill='red', outline='black')
    draw.text((70, 90), f"Carro {i+1}", fill="white")
    img.save(filename)
    print(f"Creado: {filename}")

# 5 círculos azules (simulando "Frutas")
for i in range(5):
    filename = os.path.join(output_dir, f"fruta{i+1}.jpg")
    img = Image.new('RGB', (200, 200), color = 'white')
    draw = ImageDraw.Draw(img)
    draw.ellipse(((25, 25), (175, 175)), fill='blue', outline='black')
    draw.text((70, 90), f"Fruta {i+1}", fill="white")
    img.save(filename)
    print(f"Creado: {filename}")

# 5 triángulos verdes (simulando "Personas")
for i in range(5):
    filename = os.path.join(output_dir, f"persona{i+1}.jpg")
    img = Image.new('RGB', (200, 200), color = 'white')
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 25), (25, 175), (175, 175)], fill='green', outline='black')
    draw.text((60, 90), f"Persona {i+1}", fill="white")
    img.save(filename)
    print(f"Creado: {filename}")

print("¡Listo! 15 imágenes creadas.")