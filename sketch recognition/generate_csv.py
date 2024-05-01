import os

walk = os.walk("./dataset/val/images")

for idx,file in enumerate(walk):
    __,__,filesList = file
    with open("./test_labels.csv",'w') as f:
        for name in filesList:
            name = name.split('.')[0]
            print(name)
            str = f"{name},{1}\n"
            f.write(str)

        
        

