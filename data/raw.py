import os

datasets = 'Ztree-Bark'
path = f'{datasets}/{datasets}'

listdir = os.listdir(path)
for i, x in enumerate(listdir):
    os.rename(f'{datasets}/rawData/{x}', f'{datasets}/rawData/{i}')
