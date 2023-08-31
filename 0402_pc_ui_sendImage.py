import gi
import serial
import codecs
import numpy as np
import array
from PIL import Image

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from random import randrange

from utils.dataset import MnistData


class MyWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="MNIST STM32F7 TEST UI")

        self.box = Gtk.Box(spacing=60)
        self.add(self.box)

        self.button1 = Gtk.Button(label="Random")
        self.button1.connect("clicked", self.on_button1_clicked)
        self.box.pack_start(self.button1, True, True, 0)

        self.button2 = Gtk.Button(label="Send")
        self.button2.connect("clicked", self.on_button2_clicked)
        self.box.pack_start(self.button2, True, True, 0)

        self.button3 = Gtk.Button(label="Recive")
        self.button3.connect("clicked", self.on_button3_clicked)
        self.box.pack_start(self.button3, True, True, 0)

        self.image = Gtk.Image()
        
        self.testimage = None
        self.data = MnistData('./data')
        self.testimage = None
        self.testlabel = None
        # img, label = self.data.__getitem__(0)
        

        self.box.pack_start(self.image, True, True, 0)

        self.ser = serial.Serial(port="/dev/ttyACM0", dsrdtr=0, rtscts=0)
        self.ser.baudrate = 115200
        self.sr = codecs.getreader("UTF-8")(self.ser, errors='replace')

    def swrite(self, ser, str):
        self.ser.write((str+'\n').encode('utf-8'))

    def sread(self, ser):
        return self.ser.readline().decode('utf-8')


    def on_button1_clicked(self, widget):
        index = randrange(50000)
        img, label = self.data.__getitem__(index)
        self.testimage = np.array(img).flatten()
        self.testlabel = label

        print("Select image index: %d\n" %index)
        print("Label is %d\n" %label)

        imgrgb = Image.new("RGB", img.size)
        imgrgb.paste(img)
        arr = np.array(imgrgb)
        arr = array.array('B', arr.tobytes())
        self.sendimage = arr
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(arr, GdkPixbuf.Colorspace.RGB, False, 8, 28, 28, 28*3)
        self.image.set_from_pixbuf(pixbuf)
    
    def on_button2_clicked(self, widget):
        dsize = self.testimage.size
        print("dsize: %d" %dsize)
        self.swrite(self.ser, str(dsize))
        ret = self.sread(self.ser)
        if int(ret) == dsize: print("Verification OK: %d" %int(ret))
        else: print("not pass verification")

        for i in range(dsize):
            self.swrite(self.ser, str(self.testimage[i]))    
        print("Send Over")

    def on_button3_clicked(self, widget):
        try:
            ret = self.sread(self.ser)
            print("recive %s" %(ret))
        except:
            print("No recive")
        # print("Recive: ",self.sr.readline())   


def main():
    win = MyWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()