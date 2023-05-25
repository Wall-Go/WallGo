CC = gcc
LD = gcc
CP = g++
FF=gfortran
CFLAGS =  -I/opt/homebrew/opt/libaec/include -I/opt/homebrew/include -fopenmp -fno-stack-protector -Ofast -fcommon -std=c++11 -O3 -fomit-frame-pointer -I./src/common -I.
# -march=native
LDFLAGS= -I/opt/homebrew/opt/libomp/include -lgsl -lgslcblas
LIBFLAGS=-L/opt/homebrew/lib
FFLAGS=-Wall -O -fbounds-check
RM = /bin/rm -f
OBJS = main.o FourVector.o kinematics.o CollElem.o operators.o
EXECUTABLE = collision
all:$(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CP) $(LIBFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDFLAGS) -L/usr/local/lib

operators.o: operators.cpp
	$(CC) $(CFLAGS) -c operators.cpp

main.o: main.cpp kinematics.cpp FourVector.cpp CollElem.cpp operators.cpp
	$(CP) $(LIBFLAGS) $(CFLAGS) -c main.cpp $(LDFLAGS)

clean:
	$(RM) $(EXECUTABLE) $(OBJS)
