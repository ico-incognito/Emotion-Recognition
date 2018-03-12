import uuid
import csv

emotion = []
pixels = []

with open('fer2013/fer.csv', 'r', encoding="utf8") as csv_read_file:

    # reader object
    csv_reader = csv.DictReader(csv_read_file)

    for line in csv_reader:
        emotion.append(line['emotion'])
        pixels.append(line['pixels'])

print(emotion)

print(pixels)