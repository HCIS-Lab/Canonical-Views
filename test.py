import os
from collections import Counter


classnames = [
    # "airplane,aeroplane,plane",
    # "ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin",
    # "bag,traveling bag,travelling bag,grip,suitcase",
    # "basket,handbasket",
    # "bathtub,bathing tub,bath,tub",
    # "bed",
    # "bench",
    # "birdhouse",
    # "bookshelf",
    # "bottle",
    # "bowl",
    # "bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi",
    # "cabinet",
    # "camera,photographic camera",
    # # "can,tin,tin can",
    # #"cap",
    "car,auto,automobile,machine,motorcar",
    "chair",
    "clock",
    "computer keyboard,keypad",
    "dishwasher,dish washer,dishwashing machine",
    "display,video display",
    "earphone,earpiece,headphone,phone",
    "faucet,spigot",
    "file,file cabinet,filing cabinet",
    "guitar",
    # "helmet",
    # "jar",
    # "knife",
    "lamp",
    "laptop,laptop computer",
    "loudspeaker,speaker,speaker unit,loudspeaker system,speaker system",
    "mailbox,letter box",
    "motorcycle,bike",
    "mug",
    "grand piano,grand",
    "pistol,handgun,side arm,shooting iron",
    "pot,flowerpot",
    "table",
    "telephone,phone,telephone set",
    "tower",
    "train,railroad train",
    # "vessel,watercraft",
    # "washer,automatic washer,washing machine"
    ]

def count_prefixes(folder_path):
    prefixes = []
    for fname in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, fname)):
            prefix = fname.split('_')[0]  # take part before first '_'
            if prefix not in prefixes:
                prefixes.append(prefix)

    return len(prefixes)



# Example usage:
root = "/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_40_60/"
for name in classnames:
    folder = os.path.join(root, name, 'train')
    counts = count_prefixes(folder)
    if counts <109:
        print(name)
        print(counts)
        print('----')


#59
# airplane,aeroplane,plane
# 50
# ----
# ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin
# 50
# ----
# bag,traveling bag,travelling bag,grip,suitcase
# 29
# ----
# basket,handbasket
# 31
# ----
# bathtub,bathing tub,bath,tub
# 50
# ----
# bed
# 50
# ----
# bench
# 50
# ----
# birdhouse
# 50
# ----
# bookshelf
# 50
# ----
# bottle
# 50
# ----
# bowl
# 11
# ----
# bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi
# 44
# ----
# cabinet
# 50
# ----
# camera,photographic camera
# 27
# ----
# car,auto,automobile,machine,motorcar
# 50
# ----
# chair
# 50
# ----
# clock
# 50
# ----
# computer keyboard,keypad
# 30
# ----
# dishwasher,dish washer,dishwashing machine
# 48
# ----
# display,video display
# 50
# ----
# earphone,earpiece,headphone,phone
# 10
# ----
# faucet,spigot
# 50
# ----
# file,file cabinet,filing cabinet
# 50
# ----
# guitar
# 6
# ----
# lamp
# 50
# ----
# laptop,laptop computer
# 50
# ----
# loudspeaker,speaker,speaker unit,loudspeaker system,speaker system
# 50
# ----
# mailbox,letter box
# 50
# ----
# motorcycle,bike
# 15
# ----
# mug
# 50
# ----
# grand piano,grand
# 50
# ----
# pistol,handgun,side arm,shooting iron
# 50
# ----
# pot,flowerpot
# 50
# ----
# table
# 50
# ----
# telephone,phone,telephone set
# 50
# ----
# tower
# 15
# ----
# train,railroad train
# 38
# ----




# 49
# bag,traveling bag,travelling bag,grip,suitcase
# 29
# ----
# basket,handbasket
# 31
# ----
# bowl
# 11
# ----
# bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi
# 44
# ----
# camera,photographic camera
# 27
# ----
# computer keyboard,keypad
# 30
# ----
# dishwasher,dish washer,dishwashing machine
# 48
# ----
# earphone,earpiece,headphone,phone
# 10
# ----
# motorcycle,bike
# 15
# ----
# tower
# 15
# ----
# train,railroad train
# 38
# ----

# 29
# motorcycle,bike
# 15
# ----
# tower
# 15
# ----
# train,railroad train
# 38
# ----
# (hank3d) kung@thundurus:/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection$ python test.py
# bowl
# 11
# ----
# camera,photographic camera
# 27
# ----
# earphone,earpiece,headphone,phone
# 10
# ----
# guitar
# 6
# ----
# motorcycle,bike
# 15
# ----
# tower
# 15
# ----

# 39
# bag,traveling bag,travelling bag,grip,suitcase
# 29
# ----
# basket,handbasket
# 31
# ----
# bowl
# 11
# ----
# camera,photographic camera
# 27
# ----
# computer keyboard,keypad
# 30
# ----
# earphone,earpiece,headphone,phone
# 10
# ----
# guitar
# 6
# ----
# motorcycle,bike
# 15
# ----
# tower
# 15
# ----
# train,railroad train
# 38
# ----


