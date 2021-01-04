OBJS	= multiple_reduce.o 
SOURCE	= single_reduce.cpp multiple_reduce.cpp
HEADER	= types.h utils.h activations/leakyrelu.h activations/relu.h layers/conv.h layers/net.h layers/linear.h layers/layer.h
OUT		= MPINet
CC	 	= mpic++
FLAGS	= -g -c -Wall -I/usr/include/eigen3/ -O3
LFLAGS	= -lm

all: clean $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

single_reduce.o: single_reduce.cpp
	$(CC) $(FLAGS) single_reduce.cpp -std=c++14

multiple_reduce.o: multiple_reduce.cpp
	$(CC) $(FLAGS) multiple_reduce.cpp -std=c++14

test.o: clean test.cpp
	$(CC) $(FLAGS) test.cpp -std=c++14

.PHONY: clean
clean:
	rm -f $(OBJS) $(OUT)
