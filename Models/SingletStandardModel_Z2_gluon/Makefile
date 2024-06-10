.PHONY=clean, run_main

PYTHON=python3
WOLFRAMSCRIPT=wolframscript -file
MATRIXELEMENTS=MatrixElements
MATRIX_SCRIPT=${MATRIXELEMENTS}/matrixElements.qcd.m
MATRIX_DATA=${MATRIXELEMENTS}/MatrixElements.txt
COLLISIONS=CollisionOutput
MAIN=SingletStandardModel_Z2_gluon

# File Types (fod dependencies)
COLLISION_FILES = $(shell find ${COLLISIONS} -name 'collisions_*.hdf5')

all: run_main

${COLLISIONS}:
	echo "${COLLISIONS} folder not found. Generating collisions..."; \
	mkdir -p ${COLLISIONS}

generate_collisions: ${MATRIX_DATA} | ${COLLISIONS}
	@echo "Running collision generator"
	${PYTHON} collisionGenerator.py

${MATRIX_DATA}: ${MATRIX_SCRIPT}
	@echo "Running ${MATRIX_SCRIPT}"
	${WOLFRAMSCRIPT} ${MATRIX_SCRIPT}

run_main: ${COLLISION_FILES} | generate_collisions
	@echo "Running model file"
	${PYTHON} ${MAIN}.py

model:
	@echo "Running model file"
	${PYTHON} ${MAIN}.py

generateMatrixElements: ${MATRIX_DATA}

clean:
	rm -rf ${COLLISIONS}
	rm -rf ${MATRIX_DATA}
