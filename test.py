from initializeData import *
ini = IniDataset(7)
ini.setEdgeWeight()
for i in [1, 2]:
    for j in [1, 2]:
        print(i, j)
        ini = IniWallet(7, i, j)
        ini.setNodeWallet()