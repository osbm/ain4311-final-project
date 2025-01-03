import os
import random
import uuid


# take photos from group1_photos and group2_photos and create a new dataset

group1_photos = "group1_photos/"

group2_photos = "group2_photos/"

target_train_folder = "finetuning_train/"
target_val_folder = "finetuning_val/"


all_cats = []
all_dogs = []
all_women = []
all_men = []

all_cats.extend([os.path.join(group1_photos, "cat", img) for img in os.listdir(os.path.join(group1_photos, "cat"))])
all_cats.extend([os.path.join(group2_photos, "cat", img) for img in os.listdir(os.path.join(group2_photos, "cat"))])

all_dogs.extend([os.path.join(group1_photos, "dog", img) for img in os.listdir(os.path.join(group1_photos, "dog"))])
all_dogs.extend([os.path.join(group2_photos, "dog", img) for img in os.listdir(os.path.join(group2_photos, "dog"))])

all_women.extend([os.path.join(group1_photos, "woman", img) for img in os.listdir(os.path.join(group1_photos, "woman"))])
all_women.extend([os.path.join(group2_photos, "woman", img) for img in os.listdir(os.path.join(group2_photos, "woman"))])

all_men.extend([os.path.join(group1_photos, "man", img) for img in os.listdir(os.path.join(group1_photos, "man"))])
all_men.extend([os.path.join(group2_photos, "man", img) for img in os.listdir(os.path.join(group2_photos, "man"))])


# print(type(all_cats))
# exit()
rng = random.Random(42)

os.makedirs(target_train_folder, exist_ok=True)

os.makedirs(target_val_folder, exist_ok=True)


for class_name, images in zip(["cat", "dog", "woman", "man"], [all_cats, all_dogs, all_women, all_men]):
    os.makedirs(f"{target_train_folder}{class_name}", exist_ok=True)
    os.makedirs(f"{target_val_folder}{class_name}", exist_ok=True)
    rng.shuffle(images)

    num_images = len(images)
    train_val_split = int(num_images * 0.95)

    train_images = images[:train_val_split]
    val_images = images[train_val_split:]

    print("Class Name: ", class_name)
    print("Train Images: ", len(train_images))
    print("Val Images: ", len(val_images))

    # start train

    for image in train_images:
        image_extension = image.split(".")[-1]
        image_folder = ''.join(image.split("/")[-2:])
        image_name = f"{image.split('/')[-1]}.{image_extension}"
        new_image_name = f"{uuid.uuid4()}.{image_extension}"
        os.system(f"cp '{image}' '{target_train_folder}{class_name}/{new_image_name}'")

    # start val

    for image in val_images:
        image_extension = image.split(".")[-1]
        image_folder = ''.join(image.split("/")[-2:])
        image_name = f"{image.split('/')[-1]}.{image_extension}"
        new_image_name = f"{uuid.uuid4()}.{image_extension}"
        os.system(f"cp '{image}' '{target_val_folder}{class_name}/{new_image_name}'")

# end of data/create-finetuning-dataset.py

