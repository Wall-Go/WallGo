.PHONY=clean

PYTHON=python3.11
WOLFRAMSCRIPT=wolframscript -file
MATRIXELEMENTS=MatrixElements
MATRIX_SCRIPT=${MATRIXELEMENTS}/matrixElements.ew.m
MATRIX_DATA=${MATRIXELEMENTS}/MatrixElements.txt
COLLISIONS=CollisionOutput
MAIN=TwoHiggsDoubletModel

# File Types (fod dependencies)
COLLISION_FILES = $(shell find ${COLLISIONS} -name 'collisions_*.hdf5')

all: model

${COLLISIONS}:
	echo "${COLLISIONS} folder not found. Generating collisions..."; \
	mkdir -p ${COLLISIONS} 

${MATRIX_DATA}: ${MATRIX_SCRIPT}
	@echo "Running ${MATRIX_SCRIPT}"
	${WOLFRAMSCRIPT} ${MATRIX_SCRIPT}

model: ${MATRIX_DATA}
	@echo "Running model file"
	${PYTHON} ${MAIN}.py

generateMatrixElements: ${MATRIX_DATA}

clean:
	rm -rf ${COLLISIONS}
	rm -rf ${MATRIXELEMENTS}.*${MATRIX_DATA}