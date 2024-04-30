.PHONY:

PYTHON=python3
WOLFRAMSCRIPT=wolframscript -file
MATRIX_SCRIPT=MatrixElements/MatrixElements.qcd.m
MATRIX_DATA=MatrixElements/MatrixElements.txt
MAIN=SingletStandardModel_Z2

all: run_main generate_collisions

CollisionOutput:
	mkdir -p CollisionOutput

generate_collisions: ${MATRIX_DATA} | CollisionOutput
	@if [ ! -d CollisionOutput ]; then \
		echo "CollisionOutput folder not found. Generating collisions..."; \
		mkdir CollisionOutput; \
	else \
		echo "CollisionOutput folder found."; \
	fi
	python3 collisionGenerator.py

${MATRIX_DATA}: ${MATRIX_SCRIPT}
	@echo "Running MatrixElements.m..."
	${WOLFRAMSCRIPT} ${MATRIX_SCRIPT}

run_main: generate_collisions
	@echo "Running model file"
	python3 ${MAIN}.py

clean: