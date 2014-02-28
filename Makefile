CC = g++
#MPICC = mpic++
MPICC = g++
CFLAGS = -Wall -ansi -pedantic -fopenmp -O3 -funroll-all-loops -g 
LDFLAGS = -lboost_program_options-mt -fopenmp -g
OBJECTS = Dataset.o Model.o Misc.o RBMCF.o RBMBASIC.o RBMCF_OPENMP.o  
EXEC = RunDBNCF RunRBMCF #RunDBN RunRBM

all: $(EXEC)

RunRBMCF: RunRBMCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

RunDBN: RunDBNCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(MPICC) $(CFLAGS) -o $@ -c $<

.PHONY: clean edit rebuild

clean:
	rm -f *.o *~ $(EXEC)

edit:
	geany *.h *.cpp &

rebuild: clean all
