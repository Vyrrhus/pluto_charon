################################################
#	COMPILER
################
F90		:= gfortran
FFLAGS	:= -g -O0 -Wall -Wtabs -fbacktrace -fbounds-check -ffpe-trap=zero,overflow,invalid
LFLAGS	:= -L ./fortran/src	# Répertoires des bibliothèques supplémentaires
LIBS    := -lfreqAnalys -lgfortran -lm -lpthread

# DIRECTORIES
BINDIR := bin

# SOURCES
MAIN_SRC := fortran/frequencyMapAnalysis.f90

# EXECUTABLES
MAIN_EXE := frequency_analysis

# CREATE OBJECTS
MAIN_OBJ := $(patsubst %,$(BINDIR)/%,$(notdir $(MAIN_SRC:.f90=.o)))
FFLAGS += -J $(BINDIR)

ifneq ($(BINDIR),)
$(shell test -d $(BINDIR) || mkdir -p $(BINDIR))
endif

################################################
#	TARGETS
################

# PROGRAM
default: $(MAIN_EXE)

$(MAIN_EXE): $(MAIN_OBJ) fortran/src/libfreqAnalys.a 
	$(F90) $(FFLAGS) $(LFLAGS) $^ -o $@ $(LIBS)

$(BINDIR)/%.o: fortran/%.f90
	$(F90)	$(FFLAGS) -c -o $@ $<

# CLEAN
clean:
	@rm -rf $(BINDIR)/*.p $(BINDIR)/*.mod
	@rm -rf $(MAIN_EXE)
	@echo "All binaries deleted"
	@echo "Executable deleted"
	@echo $(MAIN_OBJ)