CC = g++
#MPICC = mpic++
MPICC = g++
CFLAGS = -Wall -ansi -pedantic -fopenmp -O0 -funroll-all-loops -g -fPIC 
LDFLAGS = -lboost_program_options-mt -fopenmp -g -fPIC
OBJECTS = Dataset.o Dumb.o Ensemble.o Model.o RBM.o Misc.o RBMCF.o RBMBASIC.o RBMCF_OPENMP.o DBNCF.o DAECF.o AHRBMCF.o
EXEC = RunDBNCF RunRBMCF RunDAECF RunAHRBMCF RunDBNCF.so RunRBMCF.so RunDAECF.so RunAHRBMCF.so #RunDBN RunRBM

all: $(EXEC)

RunRBMCF: RunRBMCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

RunAHRBMCF: RunAHRBMCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

RunDBNCF: RunDBNCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

RunDAECF: RunDAECF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

RunRBMCF.so: RunRBMCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS) -shared

RunAHRBMCF.so: RunAHRBMCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS) -shared

RunDBNCF.so: RunDBNCF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS) -shared

RunDAECF.so: RunDAECF.o $(OBJECTS)
	$(MPICC) -o $@ $^ $(LDFLAGS) -shared


%.o: %.cpp
	$(MPICC) $(CFLAGS) -o $@ -c $<

.PHONY: clean edit rebuild

clean:
	rm -f *.o *~ $(EXEC)

edit:
	geany *.h *.cpp &

rebuild: clean all
