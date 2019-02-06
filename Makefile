CXX		  := g++
CXX_FLAGS :=  -std=c++11  -fopenmp -O3 

BIN		:= bin
SRC		:= src
INCLUDE	:= C:\winbuld\SFML\include -IC:\winbuld
LIB		:= C:\winbuld\SFML\lib -lsfml-graphics -lsfml-system -lsfml-window -LC:\winbuld\CL\lib -lopencl

LIBRARIES	:=
EXECUTABLE	:= main.exe
EXTCODE 	:= opencl_kernel.cl 


all: $(BIN)/$(EXECUTABLE) 
	 

run: clean all
	copy  .\$(SRC)\$(EXTCODE) .\$(BIN)\$(EXTCODE)
	cd .\$(BIN) && .\$(EXECUTABLE)
	

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE)  $^ -o $@ $(LIBRARIES) -L$(LIB)

clean:
	del /q $(BIN)\$(EXECUTABLE)
	del /q $(BIN)\$(EXTCODE)
