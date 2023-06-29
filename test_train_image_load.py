from PIL import Image


IMAGE_PATH = '/media/br/cityscapes_data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'

image = Image.open(IMAGE_PATH)
dummy_image = Image.new('RGB', [2048, 1024])


print(image)

print(*image.size)
print(dummy_image)

