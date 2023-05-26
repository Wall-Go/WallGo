#CC = gcc
#LD = gcc
#CP = g++
#FF=gfortran

CXX := g++ -std=c++11

## TODO fix the include and library paths

CXXFLAGS := -I/opt/homebrew/opt/libaec/include -I/opt/homebrew/include -fno-stack-protector -fcommon -std=c++11 -O3 -fomit-frame-pointer -I./src/common -I. -I /usr/include/hdf5/serial



SRC_DIR := ./
BUILD_DIR := ./build
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))


# -march=native
LDFLAGS := -lgsl -lgslcblas -lhdf5
LIBFLAGS := -L/opt/homebrew/lib -L /usr/lib/x86_64-linux-gnu/hdf5/serial 
FFLAGS := -Wall -O -fbounds-check

EXECUTABLE := collision

TARGET := $(BUILD_DIR)/$(EXECUTABLE)


$(TARGET): $(OBJ_FILES)
	@echo "Linking..."
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBFLAGS)

clean:
	$(RM) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d

## Build rule for objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<


all: $(TARGET)


## store object dependency graphs in .d files
-include $(OBJ_FILES:.o=.d)
