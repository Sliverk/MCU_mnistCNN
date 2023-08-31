import numpy as np
import array
from utils.dataset import MnistData
from random import randrange
import serial


def getMnistData():
    data = MnistData('./data')
    index = randrange(50000)
    img, label = data.__getitem__(index)
    return img, label


def setSerial(port="/dev/ttyACM0"):
    ser = serial.Serial(port=port, dsrdtr=0, rtscts=0)
    ser.baudrate = 115200
    return ser

def swrite(ser, str):
    ser.write((str+'\n').encode('utf-8'))

def sread(ser):
    return ser.readline().decode('utf-8')


def verifiedSend(data, ser):
    dsize = data.size
    print("dsize: %d" %dsize)
    swrite(ser, str(dsize))
    ret = sread(ser)
    if int(ret) == dsize: print("Verification OK: %d" %int(ret))
    else: print("not pass verification")

    for i in range(dsize):
        swrite(ser, str(data[i]))
        # ret = sread(ser)
        # print("recive: %s" %ret)
    print("Send Over")
    ret = sread(ser)
    print(f"recive %s" %(ret))



# input & verify
# first: number of input
# Then: one number each time with verification
def main():
    ser = setSerial("/dev/ttyACM2")

    img, label = getMnistData()
    metadata = np.array(img).flatten()
    print("Label is %d" %label)
    verifiedSend(data=metadata, ser=ser)


if __name__ == '__main__':
    main()


