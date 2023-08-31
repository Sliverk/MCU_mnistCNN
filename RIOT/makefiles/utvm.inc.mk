_CURDIR := $(shell basename $(CURDIR))
UTVM_NAME ?= $(_CURDIR)
UTVM_DIR_BASE ?= $(BINDIR)/utvm
UTVM_MODEL_DIR = $(UTVM_DIR_BASE)/$(UTVM_NAME)

all:
	$(QQ)"$(MAKE)" -C $(UTVM_MODEL_DIR)/codegen/host/src -f $(RIOTBASE)/makefiles/utvm/Makefile.utvm \
		UTVM_MODULE_NAME=$(UTVM_NAME) UTVM_NAME=$(UTVM_NAME) UTVM_MODEL_DIR=$(UTVM_MODEL_DIR)

prepare: $(CURDIR)/$(UTVM_NAME).tar
	mkdir -p $(UTVM_MODEL_DIR)
	tar --extract --file=$< --directory $(UTVM_MODEL_DIR) --touch

clean:
	$(QQ)rm -Rf $(UTVM_MODEL_DIR)
