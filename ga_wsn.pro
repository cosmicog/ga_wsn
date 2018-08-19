#-------------------------------------------------
#
# Project created by QtCreator 2014-12-28T22:35:17
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ga_wsn
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
            calculations.h

FORMS    += mainwindow.ui


OTHER_FILES += cuda_interface.cu
OTHER_FILES += testPlot.m
OTHER_FILES += nodes.txt
OTHER_FILES += myaa.m

# project build directories
DESTDIR = $$system(cd)
OBJECTS_DIR = $$DESTDIR/Obj
# and C/C++ flags
QMAKE_CFLAGS_RELEASE =-O3
QMAKE_CXXFLAGS_RELEASE =-O3
# cuda source
CUDA_SOURCES += cuda_interface.cu
# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib
# GPU architecture
CUDA_ARCH = sm_20
# NVCC flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
# Path to libraries
LIBS += -lcudart -lcuda
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS     ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

# MATLAB
INCLUDEPATH += /ogh/MATLAB/R2013b/extern/include
QMAKE_CFLAGS += -O3 -I /ogh/MATLAB/R2013b/extern/include -ansi -D_GNU_SOURCE -f /usr/local/MATLAB/R2013a/bin/engopts.sh
LIBS += -L"/ogh/MATLAB/R2013b/bin/glnxa64" -leng -lmat -lmex -lmx -Wl,-rpath=/ogh/MATLAB/R2013b/bin/glnxa64 -lfreetype
