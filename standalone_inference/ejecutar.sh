#!/bin/bash
# Script para ejecutar el modelo con el comando especificado
python test_model.py --checkpoint "novTest/alphabetnet.pt" --thresholds "novTest/thresholds.json" --regex "((A+B+((C.D)+E)*)"

