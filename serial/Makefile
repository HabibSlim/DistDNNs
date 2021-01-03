OBJS	= main.o
SOURCE	= main.cpp
HEADER	= types.h utils.h activations/leakyrelu.h activations/relu.h layers/conv.h layers/net.h layers/linear.h layers/layer.h
OUT	= MPINet
CC	 = g++
FLAGS	 = -g -c -Wall -I/usr/include/eigen3/ -O3
LFLAGS	 = -lm

all: clean $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.cpp
	$(CC) $(FLAGS) main.cpp -std=c++14

test.o: clean test.cpp
	$(CC) $(FLAGS) test.cpp -std=c++14

.PHONY: clean
clean:
	rm -f $(OBJS) $(OUT)
