import os

source = 'group1_photos/'

target_dog = 'group1_photos/dog/'
target_cat = 'group1_photos/Cat/'
target_man = 'group1_photos/Man/'
target_woman = 'group1_photos/woman/'

for file in os.listdir(source):
    if os.path.isdir(file):
        print("Directory: ", file)
        continue
    if "dog" in file:
        print("Dog: ", file)
        os.rename(source + file, target_dog + file)
    elif "cat" in file:
        print("Cat: ", file)
        os.rename(source + file, target_cat + file)
    elif "Human_Male" in file:
        print("Human Male: ", file)
        os.rename(source + file, target_man + file)
    elif "Human_Female" in file:
        print("Human female: ", file)
        os.rename(source + file, target_woman + file)
    else:
        print("Unknown file: ", file)

print("Done!")