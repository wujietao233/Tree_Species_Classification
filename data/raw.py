import os

datasets = 'Bark-Combination-88'
path = f'{datasets}/{datasets}'

listdir = os.listdir(path)
for i, x in enumerate(listdir):
    os.rename(f'{datasets}/rawData/{x}', f'{datasets}/rawData/{i}')
