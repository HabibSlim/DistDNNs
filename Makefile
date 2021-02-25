CMP	= runmpi runserial test

# Experiments
serial:
	make -f ./make/makefile.serial

param_avg: # -> Multiple parameter averaging
	make -f ./make/makefile.mpi param_avg

# Test target
test:
	make -f ./make/makefile.serial test

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(CMP)
