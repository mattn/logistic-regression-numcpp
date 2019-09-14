SRCS = \
	main.cxx

OBJS = $(subst .cc,.o,$(subst .cxx,.o,$(subst .cpp,.o,$(SRCS))))

CXXFLAGS = -I NumCpp/include
LIBS = 
TARGET = logistic-regression-numcpp
ifeq ($(OS),Windows_NT)
TARGET := $(TARGET).exe
endif

.SUFFIXES: .cpp .cxx .o

all : $(TARGET)

$(TARGET) : $(OBJS)
	g++ -std=c++17 -o $@ $(OBJS) $(LIBS)

.cxx.o :
	g++ -std=c++17 -c $(CXXFLAGS) -I. $< -o $@

.cpp.o :
	g++ -std=c++17 -c $(CXXFLAGS) -I. $< -o $@

clean :
	rm -f *.o $(TARGET)
