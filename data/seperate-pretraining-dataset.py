import os

import random
import uuid

male_path = "/home/osbm/Documents/git/ain4311-final-project/data/man"
female_path = "/home/osbm/Documents/git/ain4311-final-project/data/woman"
cat_path = "/home/osbm/Documents/git/ain4311-final-project/data/Cat"
dog_path = "/home/osbm/Documents/git/ain4311-final-project/data/Dog"
rng = random.Random(42)

for class_name, class_folder in zip(["male", "female", "cat", "dog"], [male_path, female_path, cat_path, dog_path]):
    images = os.listdir(class_folder)
    images = [os.path.join(class_folder, img) for img in images]
    rng.shuffle(images)
    num_images = len(images)
    train_images = images[:8000]
    val_images = images[8000:9300]
    for split, images in zip(["train", "val"], [train_images, val_images]):
        if os.path.exists(f"data/{split}/{class_name}"):
            continue
        os.makedirs(f"data/{split}/{class_name}", exist_ok=False)
        for image in tqdm(images, desc=f"Copying {split} images for {class_name}"):
            # give the images a unique name
            image_extension = image.split(".")[-1]
            image_folder = ''.join(image.split("/")[-2:])
            image_name = f"{image.split('/')[-1]}.{image_extension}"
            new_image_name = f"{uuid.uuid4()}.{image_extension}"
            # original image path may contain spaces, so we need to wrap it in quotes
            os.system(f"cp '{image}' 'data/{split}/{class_name}/{new_image_name}'")
