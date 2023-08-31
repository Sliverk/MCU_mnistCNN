RIOTBASE= ./RIOT

BOARD ?= stm32f746g-disco
APPLICATION = MNIST

# WERROR ?= 0
# DEVELHELP ?= 1

EXTERNAL_PKG_DIRS += pkg

USEPKG += mnistCNN 
USEMODULE += stdin

include $(RIOTBASE)/Makefile.include

CFLAGS += -Wno-strict-prototypes 
CFLAGS += -Wno-missing-include-dirs

# IOTLAB_ARCHI_openmote-b = openmoteb

override BINARY := $(ELFFILE)

# list-ttys-json:
# 	$(Q) python $(RIOTTOOLS)/usb-serial/ttys.py --format json
