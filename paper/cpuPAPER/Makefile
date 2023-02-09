CCFLAGS=-O3 -I/usr/include/openbabel-2.0 -m64 -Wall
LDFLAGS=-lopenbabel -m64

all: evaluate_hostGridVolume myOverlap cpuPAPER 
clean:
	rm -f cpuPAPER *.o

hostVolume.o: hostVolume.cpp hostVolume.h cudaVolumeTypes.h config.h
	g++ $(CCFLAGS) -c hostVolume.cpp
hostVolume_grid.o: hostVolume.cpp hostVolume.h cudaVolumeTypes.h config.h
	g++ $(CCFLAGS) -DGRID_VOLUME -DOPENMP -fopenmp -o hostVolume_grid.o -c hostVolume.cpp
cpuPAPER.o: cpuPAPER.cpp nrOptimization.cpp cudaVolumeTypes.h config.h
	g++ $(CCFLAGS) -c cpuPAPER.cpp
cpuPAPER: cpuPAPER.o hostVolume.o inputFileReader.o inputModule.o inputPreprocessor.o transformTools.o
	g++ $(LDFLAGS) -o cpuPAPER cpuPAPER.o hostVolume.o inputFileReader.o inputModule.o inputPreprocessor.o transformTools.o

evaluate_hostGridVolume.o: hostVolume.h evaluate_hostGridVolume.cpp cudaVolumeTypes.h inputFileReader.h
	g++ $(CCFLAGS) -fopenmp -c evaluate_hostGridVolume.cpp
evaluate_hostGridVolume: hostVolume_grid.o evaluate_hostGridVolume.o inputFileReader.o inputPreprocessor.o
	g++ $(LDFLAGS) -fopenmp -o evaluate_hostGridVolume hostVolume_grid.o evaluate_hostGridVolume.o inputFileReader.o inputPreprocessor.o

myOverlap.o: myOverlap.cpp cudaVolumeTypes.h config.h
	g++ $(CCFLAGS) -c myOverlap.cpp
myOverlap: myOverlap.o hostVolume.o
	g++ $(LDFLAGS) -o myOverlap myOverlap.o hostVolume.o

inputFileReader.o: inputFileReader.cpp inputFileReader.h config_input.h
	g++ $(CCFLAGS) -c inputFileReader.cpp
inputPreprocessor.o: inputPreprocessor.cpp inputPreprocessor.h config_input.h
	g++ $(CCFLAGS) -c inputPreprocessor.cpp
inputModule.o: inputModule.cpp inputModule.h inputFileReader.h config_input.h
	g++ $(CCFLAGS) -c inputModule.cpp
transformTools.o: transformTools.cpp transformTools.h
	g++ $(CCFLAGS) -c transformTools.cpp
