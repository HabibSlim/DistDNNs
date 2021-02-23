# Experiments
serial:
	make -f ./make/makefile.serial

multiple_reduce: # Multiple reduce averaging
	make -f ./make/makefile.mpi multiple

single_reduce:   # Single reduce averaging
	make -f ./make/makefile.mpi single

.PHONY: clean
clean:
	rm *.o
