CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra -Iheaders
SRC = src
HEADERS = headers
OBJ = $(SRC)/Arena.o $(SRC)/Tensor.o $(SRC)/Grad.o $(SRC)/Model.o $(SRC)/Glue.o tests/testDriver.o


all: bin/testDriver


bin:
	mkdir -p bin


bin/testDriver: $(OBJ) | bin
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm


$(SRC)/Arena.o: $(SRC)/Arena.c $(HEADERS)/Arena.h
	$(CC) $(CFLAGS) -c $< -o $@


$(SRC)/Tensor.o: $(SRC)/Tensor.c $(HEADERS)/Tensor.h $(HEADERS)/Arena.h
	$(CC) $(CFLAGS) -c $< -o $@


$(SRC)/Grad.o: $(SRC)/Grad.c $(HEADERS)/Grad.h $(HEADERS)/Tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC)/Model.o: $(SRC)/Model.c $(HEADERS)/Model.h $(HEADERS)/Grad.h
	$(CC) $(CFLAGS) -c $< -o $@


$(SRC)/Glue.o: $(SRC)/Glue.c $(HEADERS)/Glue.h $(HEADERS)/Tensor.h
	$(CC) $(CFLAGS) -c $< -o $@


tests/testDriver.o: tests/testDriver.c $(HEADERS)/Arena.h $(HEADERS)/Tensor.h $(HEADERS)/Model.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC)/*.o bin/testDriver

run: all
	./bin/testDriver

.PHONY: all clean run