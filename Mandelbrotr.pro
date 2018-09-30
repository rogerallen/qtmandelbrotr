#QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# added glm headers & it totally mangled this file.  Just adding them to git and not
# worrying about them in here...

SOURCES += \
    main.cpp \
    window.cpp \
    input.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    window.h \
    vertex.h \
    input.h

DISTFILES += \
    README.md \
    shaders/basic_frag.glsl \
    shaders/basic_vert.glsl \
    images/side1.png

RESOURCES += \
    resources.qrc

# CUDA
# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda
# GPU architecture (ADJUST FOR YOUR GPU)
CUDA_GENCODE  = arch=compute_60,code=sm_60
# manually add CUDA sources (ADJUST MANUALLY)
CUDA_SOURCES += cuda_code.cu
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
# libs used in your code
LIBS         += -L $$CUDA_DIR/lib64 -lcudart -lcuda
cuda.commands        = $$CUDA_DIR/bin/nvcc -c -gencode $$CUDA_GENCODE -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
cuda.depend_command  = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME}
cuda.input           = CUDA_SOURCES
cuda.output          = ${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
