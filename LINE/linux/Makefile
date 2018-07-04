CC = g++
CPPFLAGS = -std=c++11 -lpthread -lgsl -lgslcblas -O3 -I/home/julius/Dokumente/Uni/HPI/Module/BP-MUE1-2018/bachelor-thesis-experiments/LINE/linux/gsl/include -L/home/julius/Dokumente/Uni/HPI/Module/BP-MUE1-2018/bachelor-thesis-experiments/LINE/linux/gsl/lib
OBJS = line concatenate normalize reconstruct

all: $(OBJS)

$(OBJS):
	$(CC) $@.cpp $(CPPFLAGS) -o $@

clean:
	rm $(OBJS)